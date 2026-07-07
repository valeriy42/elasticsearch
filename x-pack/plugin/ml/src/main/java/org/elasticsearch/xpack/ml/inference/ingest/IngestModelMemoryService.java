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
import org.elasticsearch.common.util.set.Sets;
import org.elasticsearch.gateway.GatewayService;
import org.elasticsearch.ingest.IngestMetadata;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.core.ml.action.GetTrainedModelsAction;
import org.elasticsearch.xpack.core.ml.inference.IngestModelMemoryProvider;
import org.elasticsearch.xpack.core.ml.inference.ModelAliasMetadata;
import org.elasticsearch.xpack.ml.MachineLearning;
import org.elasticsearch.xpack.ml.inference.persistence.TrainedModelProvider;

import java.util.HashSet;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Master-only service that tracks JVM heap bytes required to load trained models referenced by ingest pipelines.
 */
public class IngestModelMemoryService implements ClusterStateListener, IngestModelMemoryProvider {

    private static final Logger logger = LogManager.getLogger(IngestModelMemoryService.class);

    private final TrainedModelProvider trainedModelProvider;
    private final ThreadPool threadPool;

    private final ConcurrentHashMap<ProjectId, Set<String>> referencedModelsByProject = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, OptionalLong> globalModelSizes = new ConcurrentHashMap<>();
    private final Set<String> fetchScheduledModelIds = ConcurrentHashMap.newKeySet();
    private final AtomicReference<HeapRequirement> cachedRequirement = new AtomicReference<>(new HeapRequirement(0L, true));

    public IngestModelMemoryService(TrainedModelProvider trainedModelProvider, ThreadPool threadPool) {
        this.trainedModelProvider = trainedModelProvider;
        this.threadPool = threadPool;
    }

    @Override
    public void clusterChanged(ClusterChangedEvent event) {
        if (event.state().nodes().isLocalNodeElectedMaster() == false) {
            clearAllState();
            return;
        }

        if (event.state().blocks().hasGlobalBlock(GatewayService.STATE_NOT_RECOVERED_BLOCK)) {
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
            Set<String> removedModels = referencedModelsByProject.remove(removed);
            if (removedModels != null) {
                removedModels.forEach(this::reconcileGlobalAfterProjectDrop);
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
        return cachedRequirement.get();
    }

    private void clearAllState() {
        referencedModelsByProject.clear();
        globalModelSizes.clear();
        fetchScheduledModelIds.clear();
        recomputeHeapRequirement();
    }

    private void recomputeHeapRequirement() {
        long total = 0L;
        boolean exact = true;
        for (OptionalLong size : globalModelSizes.values()) {
            if (size.isPresent()) {
                total += size.getAsLong();
            } else {
                exact = false;
            }
        }
        cachedRequirement.set(new HeapRequirement(total, exact));
    }

    private boolean ingestOrAliasRelevant(ClusterChangedEvent event, ProjectId projectId) {
        return event.customMetadataChanged(projectId, IngestMetadata.TYPE)
            || event.customMetadataChanged(projectId, ModelAliasMetadata.NAME)
            || event.previousState().metadata().projects().get(projectId) == null;
    }

    private void refreshProjectFromFullScan(ProjectId projectId, ClusterState state) {
        Set<String> nowReferenced = IngestPipelineModelReferences.resolveReferencedModelsForProject(state, projectId);
        Set<String> perProject = referencedModelsByProject.computeIfAbsent(projectId, k -> ConcurrentHashMap.newKeySet());
        Set<String> current = new HashSet<>(perProject);
        Set<String> added = Sets.difference(nowReferenced, current);
        Set<String> removed = Sets.difference(current, nowReferenced);
        for (String modelId : removed) {
            perProject.remove(modelId);
            reconcileGlobalAfterProjectDrop(modelId);
        }
        for (String modelId : added) {
            perProject.add(modelId);
            globalModelSizes.putIfAbsent(modelId, OptionalLong.empty());
            scheduleFetchIfNeeded(modelId);
        }
        recomputeHeapRequirement();
    }

    private void reconcileGlobalAfterProjectDrop(String modelId) {
        boolean stillReferenced = referencedModelsByProject.values().stream().anyMatch(set -> set.contains(modelId));
        if (stillReferenced == false) {
            globalModelSizes.remove(modelId);
            fetchScheduledModelIds.remove(modelId);
            recomputeHeapRequirement();
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
        threadPool.executor(MachineLearning.UTILITY_THREAD_POOL_NAME)
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
        boolean stillReferenced = referencedModelsByProject.values().stream().anyMatch(set -> set.contains(modelId));
        if (stillReferenced == false) {
            globalModelSizes.remove(modelId);
            fetchScheduledModelIds.remove(modelId);
            recomputeHeapRequirement();
            return;
        }
        globalModelSizes.put(modelId, size);
        recomputeHeapRequirement();
    }

    // Visible for tests
    OptionalLong getGlobalModelSizeForTests(String modelId) {
        return globalModelSizes.get(modelId);
    }

    boolean isTrackingModelForProjectForTests(ProjectId projectId, String modelId) {
        return referencedModelsByProject.getOrDefault(projectId, Set.of()).contains(modelId);
    }
}
