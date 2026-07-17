/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.datafeed.extractor;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.lucene.search.TotalHits;
import org.elasticsearch.ElasticsearchSecurityException;
import org.elasticsearch.action.search.SearchPhaseExecutionException;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.search.ShardSearchFailure;
import org.elasticsearch.common.logging.Loggers;
import org.elasticsearch.common.logging.MockAppender;
import org.elasticsearch.core.ReleasableRef;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.suggest.Suggest;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.core.ml.job.messages.Messages;
import org.elasticsearch.xpack.ml.datafeed.LinkedClusterState;
import org.elasticsearch.xpack.ml.notifications.AnomalyDetectionAuditor;
import org.junit.After;
import org.junit.Before;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.nullValue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link DataExtractorUtils#extractLinkedClusterStates(SearchResponse)},
 * using mock or real {@link SearchResponse} and {@link SearchResponse.Clusters}.
 */
public class DataExtractorUtilsTests extends ESTestCase {

    private static final Logger LOGGER = LogManager.getLogger(DataExtractorUtils.class);

    private MockAppender mockAppender;

    @Before
    public void setUpMockAppender() throws Exception {
        mockAppender = new MockAppender("data_extractor_utils_test_appender");
        mockAppender.start();
        Loggers.addAppender(LOGGER, mockAppender);
    }

    @After
    public void tearDownMockAppender() throws Exception {
        mockAppender.stop();
        Loggers.removeAppender(LOGGER, mockAppender);
    }

    public void testCloudCredentialAuthenticationFailureShouldLogWarnAndAudit() {
        AnomalyDetectionAuditor auditor = mock(AnomalyDetectionAuditor.class);
        ElasticsearchSecurityException failure = new ElasticsearchSecurityException("invalid key", RestStatus.UNAUTHORIZED);

        DataExtractorUtils.checkForCloudCredentialSearchFailure(failure, "job-1", "datafeed-1", "key-abc", auditor);

        LogEvent logEvent = mockAppender.getLastEventAndReset();
        assertNotNull(logEvent);
        assertThat(logEvent.getLevel(), equalTo(Level.WARN));
        assertThat(logEvent.getMessage().getFormattedMessage(), containsString("job-1"));
        assertThat(logEvent.getMessage().getFormattedMessage(), containsString("datafeed-1"));
        assertThat(logEvent.getMessage().getFormattedMessage(), containsString("key-abc"));
        verify(auditor).warning("job-1", Messages.getMessage(Messages.JOB_AUDIT_DATAFEED_CPS_KEY_RUNTIME_FAILURE, "key-abc"));
    }

    public void testCloudCredentialAuthorizationFailureShouldLogWarnAndAudit() {
        AnomalyDetectionAuditor auditor = mock(AnomalyDetectionAuditor.class);
        ElasticsearchSecurityException failure = new ElasticsearchSecurityException("action denied", RestStatus.FORBIDDEN);

        DataExtractorUtils.checkForCloudCredentialSearchFailure(failure, "job-1", "datafeed-1", "key-abc", auditor);

        LogEvent logEvent = mockAppender.getLastEventAndReset();
        assertNotNull(logEvent);
        assertThat(logEvent.getLevel(), equalTo(Level.WARN));
        verify(auditor).warning("job-1", Messages.getMessage(Messages.JOB_AUDIT_DATAFEED_CPS_KEY_RUNTIME_AUTHZ_FAILURE, "key-abc"));
    }

    public void testCloudCredentialWrappedAuthenticationFailureShouldClassifyViaCauseChain() {
        AnomalyDetectionAuditor auditor = mock(AnomalyDetectionAuditor.class);
        Throwable failure = new RuntimeException(new ElasticsearchSecurityException("invalid key", RestStatus.UNAUTHORIZED));

        DataExtractorUtils.checkForCloudCredentialSearchFailure(failure, "job-1", "datafeed-1", "key-abc", auditor);

        verify(auditor).warning("job-1", Messages.getMessage(Messages.JOB_AUDIT_DATAFEED_CPS_KEY_RUNTIME_FAILURE, "key-abc"));
    }

    public void testCloudCredentialSearchPhaseAuthorizationFailureShouldLogWarnAndAudit() {
        AnomalyDetectionAuditor auditor = mock(AnomalyDetectionAuditor.class);
        ShardSearchFailure shardFailure = new ShardSearchFailure(new ElasticsearchSecurityException("action denied", RestStatus.FORBIDDEN));
        SearchPhaseExecutionException failure = new SearchPhaseExecutionException(
            "query",
            "all shards failed",
            new ShardSearchFailure[] { shardFailure }
        );

        DataExtractorUtils.checkForCloudCredentialSearchFailure(failure, "job-1", "datafeed-1", "key-abc", auditor);

        LogEvent logEvent = mockAppender.getLastEventAndReset();
        assertNotNull(logEvent);
        assertThat(logEvent.getLevel(), equalTo(Level.WARN));
        verify(auditor).warning("job-1", Messages.getMessage(Messages.JOB_AUDIT_DATAFEED_CPS_KEY_RUNTIME_AUTHZ_FAILURE, "key-abc"));
    }

    public void testCloudCredentialSecurityFailureWithoutCredentialIdShouldNotLogOrAudit() {
        AnomalyDetectionAuditor auditor = mock(AnomalyDetectionAuditor.class);
        ElasticsearchSecurityException failure = new ElasticsearchSecurityException("action denied", RestStatus.FORBIDDEN);

        DataExtractorUtils.checkForCloudCredentialSearchFailure(failure, "job-1", "datafeed-1", null, auditor);

        assertNull(mockAppender.getLastEventAndReset());
        verify(auditor, never()).warning(org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.anyString());
    }

    public void testCloudCredentialNonSecurityFailureShouldNotLogOrAudit() {
        AnomalyDetectionAuditor auditor = mock(AnomalyDetectionAuditor.class);

        DataExtractorUtils.checkForCloudCredentialSearchFailure(
            new RuntimeException("connection reset"),
            "job-1",
            "datafeed-1",
            "key-abc",
            auditor
        );

        assertNull(mockAppender.getLastEventAndReset());
        verify(auditor, never()).warning(org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.anyString());
    }

    public void testExtractLinkedClusterStates_returnsEmptyWhenClustersIsNull() {
        SearchResponse response = mock(SearchResponse.class);
        when(response.getClusters()).thenReturn(null);

        List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(response);

        assertThat(states, hasSize(0));
    }

    public void testExtractLinkedClusterStates_returnsEmptyWhenClustersIsEmpty() {
        SearchResponse response = createSearchResponse(SearchResponse.Clusters.EMPTY);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(0));
        }
    }

    public void testExtractLinkedClusterStates_mapsSuccessfulToAvailable() {
        SearchResponse.Cluster cluster = new SearchResponse.Cluster(
            "remote_1",
            "remote_index",
            false,
            SearchResponse.Cluster.Status.SUCCESSFUL,
            1,
            1,
            0,
            0,
            List.of(),
            TimeValue.timeValueMillis(50L),
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("remote_1", cluster));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(1));
            LinkedClusterState state = states.get(0);
            assertThat(state.alias(), equalTo("remote_1"));
            assertThat(state.status(), equalTo(LinkedClusterState.Status.AVAILABLE));
            assertThat(state.errorReason(), nullValue());
            assertThat(state.searchLatencyMs(), equalTo(50L));
        }
    }

    public void testExtractLinkedClusterStates_mapsRunningToAvailable() {
        SearchResponse.Cluster cluster = new SearchResponse.Cluster(
            "running_cluster",
            "remote_index",
            false,
            SearchResponse.Cluster.Status.RUNNING,
            1,
            0,
            0,
            0,
            List.of(),
            TimeValue.timeValueMillis(30L),
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("running_cluster", cluster));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(1));
            assertThat(states.get(0).alias(), equalTo("running_cluster"));
            assertThat(states.get(0).status(), equalTo(LinkedClusterState.Status.AVAILABLE));
            assertThat(states.get(0).searchLatencyMs(), equalTo(30L));
        }
    }

    public void testExtractLinkedClusterStates_mapsSkippedToSkipped() {
        SearchResponse.Cluster cluster = new SearchResponse.Cluster(
            "skipped_cluster",
            "index",
            true,
            SearchResponse.Cluster.Status.SKIPPED,
            0,
            0,
            0,
            0,
            List.of(),
            null,
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("skipped_cluster", cluster));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(1));
            assertThat(states.get(0).alias(), equalTo("skipped_cluster"));
            assertThat(states.get(0).status(), equalTo(LinkedClusterState.Status.SKIPPED));
            assertThat(states.get(0).errorReason(), nullValue());
            assertThat(states.get(0).searchLatencyMs(), equalTo(0L));
        }
    }

    public void testExtractLinkedClusterStates_mapsFailedToFailed() {
        SearchResponse.Cluster cluster = new SearchResponse.Cluster(
            "failed_cluster",
            "index",
            false,
            SearchResponse.Cluster.Status.FAILED,
            1,
            0,
            0,
            1,
            List.of(),
            null,
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("failed_cluster", cluster));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(1));
            assertThat(states.get(0).alias(), equalTo("failed_cluster"));
            assertThat(states.get(0).status(), equalTo(LinkedClusterState.Status.FAILED));
        }
    }

    public void testExtractLinkedClusterStates_extractsErrorReasonFromFirstFailure() {
        ShardSearchFailure failure = new ShardSearchFailure(new RuntimeException("index_not_found_exception: no such index"));
        SearchResponse.Cluster cluster = new SearchResponse.Cluster(
            "failed_cluster",
            "index",
            false,
            SearchResponse.Cluster.Status.FAILED,
            1,
            0,
            0,
            1,
            List.of(failure),
            null,
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("failed_cluster", cluster));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(1));
            assertThat(states.get(0).errorReason(), containsString("index_not_found_exception: no such index"));
        }
    }

    public void testExtractLinkedClusterStates_extractsSearchLatencyFromTook() {
        SearchResponse.Cluster cluster = new SearchResponse.Cluster(
            "remote_1",
            "idx",
            false,
            SearchResponse.Cluster.Status.SUCCESSFUL,
            2,
            2,
            0,
            0,
            List.of(),
            TimeValue.timeValueMillis(123L),
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("remote_1", cluster));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states.get(0).searchLatencyMs(), equalTo(123L));
        }
    }

    public void testExtractLinkedClusterStates_mapsPartialToAvailable() {
        SearchResponse.Cluster cluster = new SearchResponse.Cluster(
            "partial_cluster",
            "idx",
            false,
            SearchResponse.Cluster.Status.PARTIAL,
            2,
            1,
            0,
            1,
            List.of(),
            TimeValue.timeValueMillis(10L),
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("partial_cluster", cluster));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(1));
            assertThat(states.get(0).status(), equalTo(LinkedClusterState.Status.AVAILABLE));
        }
    }

    public void testExtractLinkedClusterStates_multipleClusters() {
        SearchResponse.Cluster local = new SearchResponse.Cluster(
            "",
            "local_idx",
            false,
            SearchResponse.Cluster.Status.SUCCESSFUL,
            1,
            1,
            0,
            0,
            List.of(),
            TimeValue.timeValueMillis(5L),
            false,
            "(local)"
        );
        SearchResponse.Cluster remote = new SearchResponse.Cluster(
            "remote_a",
            "remote_idx",
            false,
            SearchResponse.Cluster.Status.SUCCESSFUL,
            1,
            1,
            0,
            0,
            List.of(),
            TimeValue.timeValueMillis(20L),
            false,
            null
        );
        SearchResponse.Clusters clusters = new SearchResponse.Clusters(Map.of("", local, "remote_a", remote));
        SearchResponse response = createSearchResponse(clusters);

        try (var responseRef = ReleasableRef.of(response)) {
            List<LinkedClusterState> states = DataExtractorUtils.extractLinkedClusterStates(responseRef.get());
            assertThat(states, hasSize(2));
            LinkedClusterState localState = states.stream().filter(s -> "(local)".equals(s.alias())).findFirst().orElseThrow();
            LinkedClusterState remoteState = states.stream().filter(s -> "remote_a".equals(s.alias())).findFirst().orElseThrow();
            assertThat(localState.status(), equalTo(LinkedClusterState.Status.AVAILABLE));
            assertThat(localState.searchLatencyMs(), equalTo(5L));
            assertThat(remoteState.status(), equalTo(LinkedClusterState.Status.AVAILABLE));
            assertThat(remoteState.searchLatencyMs(), equalTo(20L));
        }
    }

    public void testPreferRicherLinkedClusterStates_keepsBestWhenCandidateEmpty() {
        List<LinkedClusterState> best = List.of(
            new LinkedClusterState("a", LinkedClusterState.Status.AVAILABLE, null, 1L),
            new LinkedClusterState("b", LinkedClusterState.Status.AVAILABLE, null, 2L)
        );
        assertThat(DataExtractorUtils.preferRicherLinkedClusterStates(best, List.of()), equalTo(best));
    }

    public void testPreferRicherLinkedClusterStates_prefersMoreDistinctAliases() {
        List<LinkedClusterState> one = List.of(new LinkedClusterState("a", LinkedClusterState.Status.AVAILABLE, null, 1L));
        List<LinkedClusterState> two = List.of(
            new LinkedClusterState("a", LinkedClusterState.Status.AVAILABLE, null, 1L),
            new LinkedClusterState("b", LinkedClusterState.Status.SKIPPED, "x", 0L)
        );
        assertThat(DataExtractorUtils.preferRicherLinkedClusterStates(one, two), equalTo(two));
        assertThat(DataExtractorUtils.preferRicherLinkedClusterStates(two, one), equalTo(two));
    }

    public void testPreferRicherLinkedClusterStates_tieBreaksOnAvailability() {
        List<LinkedClusterState> mix = List.of(
            new LinkedClusterState("a", LinkedClusterState.Status.AVAILABLE, null, 1L),
            new LinkedClusterState("b", LinkedClusterState.Status.SKIPPED, "x", 0L)
        );
        List<LinkedClusterState> bothUp = List.of(
            new LinkedClusterState("a", LinkedClusterState.Status.AVAILABLE, null, 1L),
            new LinkedClusterState("b", LinkedClusterState.Status.AVAILABLE, null, 1L)
        );
        assertThat(DataExtractorUtils.preferRicherLinkedClusterStates(mix, bothUp), equalTo(bothUp));
        assertThat(DataExtractorUtils.preferRicherLinkedClusterStates(bothUp, mix), equalTo(bothUp));
    }

    private static SearchResponse createSearchResponse(SearchResponse.Clusters clusters) {
        return new SearchResponse(
            new SearchHits(SearchHits.EMPTY, new TotalHits(0, TotalHits.Relation.EQUAL_TO), 0.0f),
            null,
            new Suggest(Collections.emptyList()),
            false,
            null,
            null,
            0,
            null,
            1,
            1,
            0,
            0,
            ShardSearchFailure.EMPTY_ARRAY,
            clusters
        );
    }
}
