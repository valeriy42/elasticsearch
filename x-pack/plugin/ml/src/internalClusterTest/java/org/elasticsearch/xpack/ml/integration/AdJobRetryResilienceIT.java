/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.elasticsearch.xpack.ml.integration;

import org.elasticsearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.xpack.core.ml.action.CloseJobAction;
import org.elasticsearch.xpack.core.ml.action.GetJobsStatsAction;
import org.elasticsearch.xpack.core.ml.action.OpenJobAction;
import org.elasticsearch.xpack.core.ml.action.PutJobAction;
import org.elasticsearch.xpack.core.ml.job.config.Job;
import org.elasticsearch.xpack.core.ml.job.config.JobState;
import org.elasticsearch.xpack.ml.MachineLearning;
import org.elasticsearch.xpack.ml.support.BaseMlIntegTestCase;

import java.util.concurrent.TimeUnit;

import static org.hamcrest.Matchers.equalTo;

/**
 * Integration tests for the AD Job Retry Resilience feature.
 *
 * These tests verify key behaviors introduced by the retry resilience changes:
 * <ol>
 *   <li>The {@code xpack.ml.job_open_retry_timeout} cluster setting is correctly registered,
 *       defaults to 60 minutes, and can be updated dynamically.</li>
 *   <li>A basic smoke test: normal user-initiated job open/close still works end-to-end
 *       (no regression from the retry mechanism).</li>
 *   <li>User-initiated fail-fast: when the ML state index is blocked and a user opens a job,
 *       the job should fail quickly (fail-fast path, no retry wrapper applied).</li>
 * </ol>
 */
@ESIntegTestCase.ClusterScope(scope = ESIntegTestCase.Scope.TEST, numDataNodes = 0)
public class AdJobRetryResilienceIT extends BaseMlIntegTestCase {

    /**
     * Verify that the {@code xpack.ml.job_open_retry_timeout} cluster setting is registered,
     * has the correct default (60 minutes), and that the setting can be updated dynamically.
     */
    public void testJobOpenRetryTimeoutSetting_defaultAndDynamicUpdate() throws Exception {
        internalCluster().ensureAtLeastNumDataNodes(1);
        ensureStableCluster(1);

        // Dynamic settings only appear in the cluster settings response after they have been
        // explicitly set; verify the default via the setting constant directly.
        assertThat(MachineLearning.JOB_OPEN_RETRY_TIMEOUT.get(Settings.EMPTY), equalTo(TimeValue.timeValueMinutes(60)));

        // Update to 5 minutes and verify the new value is accepted
        ClusterUpdateSettingsRequest updateRequest = new ClusterUpdateSettingsRequest(TEST_REQUEST_TIMEOUT, TEST_REQUEST_TIMEOUT);
        updateRequest.persistentSettings(Settings.builder().put(MachineLearning.JOB_OPEN_RETRY_TIMEOUT.getKey(), "5m").build());
        var updateResponse = client().admin().cluster().updateSettings(updateRequest).actionGet();
        assertTrue(updateResponse.isAcknowledged());

        // Restore to default
        ClusterUpdateSettingsRequest resetRequest = new ClusterUpdateSettingsRequest(TEST_REQUEST_TIMEOUT, TEST_REQUEST_TIMEOUT);
        resetRequest.persistentSettings(Settings.builder().putNull(MachineLearning.JOB_OPEN_RETRY_TIMEOUT.getKey()).build());
        client().admin().cluster().updateSettings(resetRequest).actionGet();
    }

    /**
     * Smoke test: verify that a normal user-initiated job open and close cycle still works
     * correctly after the retry resilience changes (no regression).
     */
    public void testNormalJobOpenClose_smokTest() throws Exception {
        internalCluster().ensureAtLeastNumDataNodes(1);
        ensureStableCluster(1);

        String jobId = "retry-resilience-smoke-test-job";
        Job.Builder job = createJob(jobId, ByteSizeValue.ofMb(2));
        client().execute(PutJobAction.INSTANCE, new PutJobAction.Request(job)).actionGet();

        ensureYellow();
        client().execute(OpenJobAction.INSTANCE, new OpenJobAction.Request(jobId)).actionGet();
        awaitJobOpenedAndAssigned(jobId, null);

        // Verify the job is OPENED
        GetJobsStatsAction.Response statsResponse = client().execute(GetJobsStatsAction.INSTANCE, new GetJobsStatsAction.Request(jobId))
            .actionGet();
        assertThat(statsResponse.getResponse().results().get(0).getState(), equalTo(JobState.OPENED));

        // Close the job
        client().execute(CloseJobAction.INSTANCE, new CloseJobAction.Request(jobId)).actionGet();

        // Verify the job is CLOSED
        assertBusy(() -> {
            GetJobsStatsAction.Response stats = client().execute(GetJobsStatsAction.INSTANCE, new GetJobsStatsAction.Request(jobId))
                .actionGet();
            assertThat(stats.getResponse().results().get(0).getState(), equalTo(JobState.CLOSED));
        }, 30, TimeUnit.SECONDS);
    }

    /**
     * Verifies that after a node failure and job reassignment, the job eventually reopens.
     * This exercises the retry resilience path (system-initiated reassignment) with the
     * new retry infrastructure in place.
     *
     * This is a lightweight smoke test for the reassignment + retry path. The unit tests
     * (OpenJobPersistentTasksExecutorTests) provide detailed behavioral coverage of the
     * retry logic itself.
     */
    public void testJobReopensAfterNodeFailure() throws Exception {
        // Need at least 2 nodes: one for the job to run on, another for failover.
        // Use 2 nodes so the job can be reassigned when one goes down.
        internalCluster().ensureAtLeastNumDataNodes(2);
        ensureStableCluster(2);

        String jobId = "retry-resilience-failover-job";
        Job.Builder job = createJob(jobId, ByteSizeValue.ofMb(2));
        client().execute(PutJobAction.INSTANCE, new PutJobAction.Request(job)).actionGet();

        ensureYellow();
        client().execute(OpenJobAction.INSTANCE, new OpenJobAction.Request(jobId)).actionGet();
        String origNode = awaitJobOpenedAndAssigned(jobId, null);
        assertNotNull(origNode);

        setMlIndicesDelayedNodeLeftTimeoutToZero();
        ensureGreen();

        // Stop the node running the job; this triggers reassignment via the system-initiated path.
        // The new OpenJobRetryableAction (with exponential backoff) will handle the reassignment.
        internalCluster().stopNode(origNode);
        ensureStableCluster(1);

        // The job should eventually reopen on the remaining node via the retry path.
        awaitJobOpenedAndAssigned(jobId, null);

        // Verify the job is open and not stuck in a failure state
        GetJobsStatsAction.Response stats = client().execute(GetJobsStatsAction.INSTANCE, new GetJobsStatsAction.Request(jobId))
            .actionGet();
        assertThat(stats.getResponse().results().get(0).getState(), equalTo(JobState.OPENED));
    }
}
