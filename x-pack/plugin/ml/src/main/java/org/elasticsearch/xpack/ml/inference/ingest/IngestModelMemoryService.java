/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.ingest;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.cluster.ClusterChangedEvent;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.ClusterStateListener;
import org.elasticsearch.cluster.metadata.ProjectId;
import org.elasticsearch.cluster.metadata.ProjectMetadata;
import org.elasticsearch.cluster.service.ClusterService;
import org.elasticsearch.common.util.set.Sets;
import org.elasticsearch.ingest.IngestMetadata;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.core.ml.action.GetTrainedModelsAction;
import org.elasticsearch.xpack.core.ml.inference.IngestModelMemoryProvider;
import org.elasticsearch.xpack.core.ml.inference.ModelAliasMetadata;
import org.elasticsearch.xpack.ml.inference.persistence.TrainedModelProvider;

import java.util.HashSet;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Master-only service that tracks JVM heap bytes required to load DFA models referenced by ingest pipelines.
 */
public class IngestModelMemoryService implements ClusterStateListener, IngestModelMemoryProvider {

    private static final Logger logger = LogManager.getLogger(IngestModelMemoryService.class);

    private final TrainedModelProvider trainedModelProvider;
    private final ThreadPool threadPool;

    private final ConcurrentHashMap<ProjectId, ConcurrentHashMap<String, OptionalLong>> modelSizesByProject = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, OptionalLong> globalModelSizes = new ConcurrentHashMap<>();
    private final Set<String> fetchScheduledModelIds = ConcurrentHashMap.newKeySet();

    public IngestModelMemoryService(ClusterService clusterService, TrainedModelProvider trainedModelProvider, ThreadPool threadPool) {
        this.trainedModelProvider = trainedModelProvider;
        this.threadPool = threadPool;
    }

    @Override
    public void clusterChanged(ClusterChangedEvent event) {
        if (event.state().nodes().isLocalNodeElectedMaster() == false) {
            clearAllState();
            return;
        }

        if (becameMaster(event)) {
            repopulateAllProjects(event.state());
            return;
        }

        handleRemovedProjects(event);
        handleAddedProjects(event);
        handleChangedProjects(event);
    }

    private static boolean becameMaster(ClusterChangedEvent event) {
        return event.localNodeMaster() && event.previousState().nodes().isLocalNodeElectedMaster() == false;
    }

    private void repopulateAllProjects(ClusterState state) {
        clearAllState();
        for (ProjectMetadata project : state.metadata().projects().values()) {
            refreshProjectFromFullScan(project.id(), state);
        }
    }

    private void handleRemovedProjects(ClusterChangedEvent event) {
        for (ProjectId removed : event.projectDelta().removed()) {
            ConcurrentHashMap<String, OptionalLong> removedMap = modelSizesByProject.remove(removed);
            if (removedMap != null) {
                removedMap.keySet().forEach(this::reconcileGlobalAfterProjectDrop);
            }
        }
    }

    private void handleAddedProjects(ClusterChangedEvent event) {
        for (ProjectId added : event.projectDelta().added()) {
            if (ingestOrAliasRelevant(event, added)) {
                refreshProjectFromFullScan(added, event.state());
            }
        }
    }

    private void handleChangedProjects(ClusterChangedEvent event) {
        for (ProjectMetadata project : event.state().metadata().projects().values()) {
            ProjectId projectId = project.id();
            if (event.projectDelta().added().contains(projectId)) {
                continue;
            }
            if (event.customMetadataChanged(projectId, IngestMetadata.TYPE)
                || event.customMetadataChanged(projectId, ModelAliasMetadata.NAME)) {
                refreshProjectFromFullScan(projectId, event.state());
            }
        }
    }

    @Override
    public HeapRequirement getRequiredHeapBytes() {
        long total = 0L;
        boolean exact = true;
        for (OptionalLong size : globalModelSizes.values()) {
            if (size.isPresent()) {
                total += size.getAsLong();
            } else {
                exact = false;
            }
        }
        return new HeapRequirement(total, exact);
    }

    private void clearAllState() {
        modelSizesByProject.clear();
        globalModelSizes.clear();
        fetchScheduledModelIds.clear();
    }

    private boolean ingestOrAliasRelevant(ClusterChangedEvent event, ProjectId projectId) {
        return event.customMetadataChanged(projectId, IngestMetadata.TYPE)
            || event.customMetadataChanged(projectId, ModelAliasMetadata.NAME)
            || event.previousState().metadata().projects().get(projectId) == null;
    }

    private void refreshProjectFromFullScan(ProjectId projectId, ClusterState state) {
        Set<String> nowReferenced = IngestPipelineModelReferences.resolveReferencedModelsForProject(state, projectId);
        ConcurrentHashMap<String, OptionalLong> perProject = modelSizesByProject.computeIfAbsent(projectId, k -> new ConcurrentHashMap<>());
        Set<String> current = new HashSet<>(perProject.keySet());
        Set<String> added = Sets.difference(nowReferenced, current);
        Set<String> removed = Sets.difference(current, nowReferenced);
        removed.forEach(modelId -> {
            perProject.remove(modelId);
            reconcileGlobalAfterProjectDrop(modelId);
        });
        for (String modelId : added) {
            perProject.put(modelId, OptionalLong.empty());
            globalModelSizes.putIfAbsent(modelId, OptionalLong.empty());
            scheduleFetchIfNeeded(modelId);
        }
        if (added.isEmpty() == false || removed.isEmpty() == false) {
            logger.info(
                "RFC 0007 ingest model references changed for project [{}]: added={}, removed={}, tracked_models={}",
                projectId,
                added,
                removed,
                perProject.size()
            );
        }
    }

    private void reconcileGlobalAfterProjectDrop(String modelId) {
        boolean stillReferenced = modelSizesByProject.values().stream().anyMatch(map -> map.containsKey(modelId));
        if (stillReferenced == false) {
            globalModelSizes.remove(modelId);
            fetchScheduledModelIds.remove(modelId);
        }
    }

    private void scheduleFetchIfNeeded(String modelId) {
        OptionalLong currentSize = globalModelSizes.get(modelId);
        if (currentSize != null && currentSize.isPresent()) {
            return;
        }
        if (fetchScheduledModelIds.add(modelId) == false) {
            return;
        }
        threadPool.executor(ThreadPool.Names.MANAGEMENT)
            .execute(
                () -> trainedModelProvider.getTrainedModel(
                    modelId,
                    GetTrainedModelsAction.Includes.empty(),
                    null,
                    ActionListener.wrap(config -> {
                        propagateModelSize(modelId, toStoredSize(config.getModelSize()));
                        fetchScheduledModelIds.remove(modelId);
                    }, e -> {
                        logger.warn("Could not fetch config for ingest model [{}]: {}", modelId, e.getMessage());
                        fetchScheduledModelIds.remove(modelId);
                    })
                )
            );
    }

    private static OptionalLong toStoredSize(long modelSizeBytes) {
        return modelSizeBytes > 0 ? OptionalLong.of(modelSizeBytes) : OptionalLong.empty();
    }

    private void propagateModelSize(String modelId, OptionalLong size) {
        globalModelSizes.put(modelId, size);
        for (ConcurrentHashMap<String, OptionalLong> perProject : modelSizesByProject.values()) {
            if (perProject.containsKey(modelId)) {
                perProject.put(modelId, size);
            }
        }
        logger.info(
            "RFC 0007 ingest model heap size resolved for model [{}]: size_bytes={}, present={}",
            modelId,
            size.isPresent() ? size.getAsLong() : 0L,
            size.isPresent()
        );
    }

    // Visible for tests
    OptionalLong getGlobalModelSizeForTests(String modelId) {
        return globalModelSizes.get(modelId);
    }

    boolean isTrackingModelForProjectForTests(ProjectId projectId, String modelId) {
        ConcurrentHashMap<String, OptionalLong> perProject = modelSizesByProject.get(projectId);
        return perProject != null && perProject.containsKey(modelId);
    }
}
