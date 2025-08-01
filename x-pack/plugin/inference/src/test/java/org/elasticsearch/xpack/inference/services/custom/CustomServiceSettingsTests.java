/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.custom;

import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.core.ml.AbstractBWCWireSerializationTestCase;
import org.elasticsearch.xpack.inference.InferenceNamedWriteablesProvider;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.ServiceFields;
import org.elasticsearch.xpack.inference.services.custom.response.CompletionResponseParser;
import org.elasticsearch.xpack.inference.services.custom.response.NoopResponseParser;
import org.elasticsearch.xpack.inference.services.custom.response.RerankResponseParser;
import org.elasticsearch.xpack.inference.services.custom.response.SparseEmbeddingResponseParser;
import org.elasticsearch.xpack.inference.services.custom.response.TextEmbeddingResponseParser;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.hamcrest.MatcherAssert;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.is;

public class CustomServiceSettingsTests extends AbstractBWCWireSerializationTestCase<CustomServiceSettings> {
    public static CustomServiceSettings createRandom(String inputUrl) {
        var taskType = randomFrom(TaskType.TEXT_EMBEDDING, TaskType.RERANK, TaskType.SPARSE_EMBEDDING, TaskType.COMPLETION);

        SimilarityMeasure similarityMeasure = null;
        Integer dims = null;
        var isTextEmbeddingModel = taskType.equals(TaskType.TEXT_EMBEDDING);
        if (isTextEmbeddingModel) {
            similarityMeasure = SimilarityMeasure.DOT_PRODUCT;
            dims = 1536;
        }
        var maxInputTokens = randomBoolean() ? null : randomIntBetween(128, 256);
        var url = inputUrl != null ? inputUrl : randomAlphaOfLength(15);
        Map<String, String> headers = randomBoolean() ? Map.of() : Map.of("key", "value");
        var queryParameters = randomBoolean()
            ? QueryParameters.EMPTY
            : new QueryParameters(List.of(new QueryParameters.Parameter("key", "value")));
        var requestContentString = randomAlphaOfLength(10);

        var responseJsonParser = switch (taskType) {
            case TEXT_EMBEDDING -> new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.FLOAT);
            case SPARSE_EMBEDDING -> new SparseEmbeddingResponseParser(
                "$.result.sparse_embeddings[*].embedding[*].token_id",
                "$.result.sparse_embeddings[*].embedding[*].weights"
            );
            case RERANK -> new RerankResponseParser(
                "$.result.reranked_results[*].index",
                "$.result.reranked_results[*].relevance_score",
                "$.result.reranked_results[*].document_text"
            );
            case COMPLETION -> new CompletionResponseParser("$.result.text");
            default -> new NoopResponseParser();
        };

        RateLimitSettings rateLimitSettings = new RateLimitSettings(randomLongBetween(1, 1000000));

        return new CustomServiceSettings(
            new CustomServiceSettings.TextEmbeddingSettings(similarityMeasure, dims, maxInputTokens),
            url,
            headers,
            queryParameters,
            requestContentString,
            responseJsonParser,
            rateLimitSettings
        );
    }

    public static CustomServiceSettings createRandom() {
        return createRandom(randomAlphaOfLength(5));
    }

    public void testFromMap() {
        String similarity = SimilarityMeasure.DOT_PRODUCT.toString();
        Integer dims = 1536;
        Integer maxInputTokens = 512;
        String url = "http://www.abc.com";
        Map<String, String> headers = Map.of("key", "value");
        var queryParameters = List.of(List.of("key", "value"));
        String requestContentString = "request body";

        var responseParser = new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.FLOAT);

        var settings = CustomServiceSettings.fromMap(
            new HashMap<>(
                Map.of(
                    ServiceFields.SIMILARITY,
                    similarity,
                    ServiceFields.DIMENSIONS,
                    dims,
                    ServiceFields.MAX_INPUT_TOKENS,
                    maxInputTokens,
                    CustomServiceSettings.URL,
                    url,
                    CustomServiceSettings.HEADERS,
                    headers,
                    QueryParameters.QUERY_PARAMETERS,
                    queryParameters,
                    CustomServiceSettings.REQUEST,
                    requestContentString,
                    CustomServiceSettings.RESPONSE,
                    new HashMap<>(
                        Map.of(
                            CustomServiceSettings.JSON_PARSER,
                            new HashMap<>(
                                Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                            )
                        )
                    ),
                    CustomServiceSettings.BATCH_SIZE,
                    11
                )
            ),
            ConfigurationParseContext.REQUEST,
            TaskType.TEXT_EMBEDDING
        );

        assertThat(
            settings,
            is(
                new CustomServiceSettings(
                    new CustomServiceSettings.TextEmbeddingSettings(SimilarityMeasure.DOT_PRODUCT, dims, maxInputTokens),
                    url,
                    headers,
                    new QueryParameters(List.of(new QueryParameters.Parameter("key", "value"))),
                    requestContentString,
                    responseParser,
                    new RateLimitSettings(10_000),
                    11,
                    InputTypeTranslator.EMPTY_TRANSLATOR
                )
            )
        );
    }

    public void testFromMap_EmbeddingType_Bit() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var responseParser = new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.BIT);

        var settings = CustomServiceSettings.fromMap(
            new HashMap<>(
                Map.of(
                    CustomServiceSettings.URL,
                    url,
                    CustomServiceSettings.REQUEST,
                    requestContentString,
                    CustomServiceSettings.RESPONSE,
                    new HashMap<>(
                        Map.of(
                            CustomServiceSettings.JSON_PARSER,
                            new HashMap<>(
                                Map.of(
                                    TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS,
                                    "$.result.embeddings[*].embedding",
                                    TextEmbeddingResponseParser.EMBEDDING_TYPE,
                                    CustomServiceEmbeddingType.BIT.toString()
                                )
                            )
                        )
                    )
                )
            ),
            ConfigurationParseContext.REQUEST,
            TaskType.TEXT_EMBEDDING
        );

        MatcherAssert.assertThat(
            settings,
            is(
                new CustomServiceSettings(
                    new CustomServiceSettings.TextEmbeddingSettings(null, null, null),
                    url,
                    Map.of(),
                    null,
                    requestContentString,
                    responseParser,
                    new RateLimitSettings(10_000)
                )
            )
        );
    }

    public void testFromMap_EmbeddingType_Binary() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var responseParser = new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.BINARY);

        var settings = CustomServiceSettings.fromMap(
            new HashMap<>(
                Map.of(
                    CustomServiceSettings.URL,
                    url,
                    CustomServiceSettings.REQUEST,
                    requestContentString,
                    CustomServiceSettings.RESPONSE,
                    new HashMap<>(
                        Map.of(
                            CustomServiceSettings.JSON_PARSER,
                            new HashMap<>(
                                Map.of(
                                    TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS,
                                    "$.result.embeddings[*].embedding",
                                    TextEmbeddingResponseParser.EMBEDDING_TYPE,
                                    CustomServiceEmbeddingType.BINARY.toString()
                                )
                            )
                        )
                    )
                )
            ),
            ConfigurationParseContext.REQUEST,
            TaskType.TEXT_EMBEDDING
        );

        MatcherAssert.assertThat(
            settings,
            is(
                new CustomServiceSettings(
                    new CustomServiceSettings.TextEmbeddingSettings(null, null, null),
                    url,
                    Map.of(),
                    null,
                    requestContentString,
                    responseParser,
                    new RateLimitSettings(10_000)
                )
            )
        );
    }

    public void testFromMap_EmbeddingType_Byte() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var responseParser = new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.BYTE);

        var settings = CustomServiceSettings.fromMap(
            new HashMap<>(
                Map.of(
                    CustomServiceSettings.URL,
                    url,
                    CustomServiceSettings.REQUEST,
                    requestContentString,
                    CustomServiceSettings.RESPONSE,
                    new HashMap<>(
                        Map.of(
                            CustomServiceSettings.JSON_PARSER,
                            new HashMap<>(
                                Map.of(
                                    TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS,
                                    "$.result.embeddings[*].embedding",
                                    TextEmbeddingResponseParser.EMBEDDING_TYPE,
                                    CustomServiceEmbeddingType.BYTE.toString()
                                )
                            )
                        )
                    )
                )
            ),
            ConfigurationParseContext.REQUEST,
            TaskType.TEXT_EMBEDDING
        );

        MatcherAssert.assertThat(
            settings,
            is(
                new CustomServiceSettings(
                    new CustomServiceSettings.TextEmbeddingSettings(null, null, null),
                    url,
                    Map.of(),
                    null,
                    requestContentString,
                    responseParser,
                    new RateLimitSettings(10_000)
                )
            )
        );

        assertThat(settings.elementType(), is(DenseVectorFieldMapper.ElementType.BYTE));
    }

    public void testFromMap_Completion_NoEmbeddingType() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var responseParser = new CompletionResponseParser("$.result.text");

        var settings = CustomServiceSettings.fromMap(
            new HashMap<>(
                Map.of(
                    CustomServiceSettings.URL,
                    url,
                    CustomServiceSettings.REQUEST,
                    requestContentString,
                    CustomServiceSettings.RESPONSE,
                    new HashMap<>(
                        Map.of(
                            CustomServiceSettings.JSON_PARSER,
                            new HashMap<>(Map.of(CompletionResponseParser.COMPLETION_PARSER_RESULT, "$.result.text"))
                        )
                    )
                )
            ),
            ConfigurationParseContext.REQUEST,
            TaskType.COMPLETION
        );

        MatcherAssert.assertThat(
            settings,
            is(
                new CustomServiceSettings(
                    new CustomServiceSettings.TextEmbeddingSettings(null, null, null),
                    url,
                    Map.of(),
                    null,
                    requestContentString,
                    responseParser,
                    new RateLimitSettings(10_000)
                )
            )
        );
        assertNull(settings.elementType());
    }

    public void testFromMap_Completion_ThrowsWhenEmbeddingIsIncludedInMap() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var exception = expectThrows(
            ElasticsearchStatusException.class,
            () -> CustomServiceSettings.fromMap(
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.URL,
                        url,
                        CustomServiceSettings.REQUEST,
                        requestContentString,
                        CustomServiceSettings.RESPONSE,
                        new HashMap<>(
                            Map.of(
                                CustomServiceSettings.JSON_PARSER,
                                new HashMap<>(
                                    Map.of(
                                        CompletionResponseParser.COMPLETION_PARSER_RESULT,
                                        "$.result.text",
                                        TextEmbeddingResponseParser.EMBEDDING_TYPE,
                                        "byte"
                                    )
                                )
                            )
                        )
                    )
                ),
                ConfigurationParseContext.REQUEST,
                TaskType.COMPLETION
            )
        );

        assertThat(
            exception.getMessage(),
            is(
                "Configuration contains unknown settings [{embedding_type=byte}] while parsing field [json_parser] "
                    + "for settings [custom_service_settings]"
            )
        );
    }

    public void testFromMap_WithOptionalsNotSpecified() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var responseParser = new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.FLOAT);

        var settings = CustomServiceSettings.fromMap(
            new HashMap<>(
                Map.of(
                    CustomServiceSettings.URL,
                    url,
                    CustomServiceSettings.REQUEST,
                    requestContentString,
                    CustomServiceSettings.RESPONSE,
                    new HashMap<>(
                        Map.of(
                            CustomServiceSettings.JSON_PARSER,
                            new HashMap<>(
                                Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                            )
                        )
                    )
                )
            ),
            ConfigurationParseContext.REQUEST,
            TaskType.TEXT_EMBEDDING
        );

        MatcherAssert.assertThat(
            settings,
            is(
                new CustomServiceSettings(
                    CustomServiceSettings.TextEmbeddingSettings.DEFAULT_FLOAT,
                    url,
                    Map.of(),
                    null,
                    requestContentString,
                    responseParser,
                    new RateLimitSettings(10_000)
                )
            )
        );
    }

    public void testFromMap_RemovesNullValues_FromMaps() {
        String similarity = SimilarityMeasure.DOT_PRODUCT.toString();
        Integer dims = 1536;
        Integer maxInputTokens = 512;
        String url = "http://www.abc.com";

        var headersWithNulls = new HashMap<String, Object>();
        headersWithNulls.put("value", "abc");
        headersWithNulls.put("null", null);

        String requestContentString = "request body";

        var responseParser = new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.FLOAT);

        var settings = CustomServiceSettings.fromMap(
            new HashMap<>(
                Map.of(
                    ServiceFields.SIMILARITY,
                    similarity,
                    ServiceFields.DIMENSIONS,
                    dims,
                    ServiceFields.MAX_INPUT_TOKENS,
                    maxInputTokens,
                    CustomServiceSettings.URL,
                    url,
                    CustomServiceSettings.HEADERS,
                    headersWithNulls,
                    CustomServiceSettings.REQUEST,
                    requestContentString,
                    CustomServiceSettings.RESPONSE,
                    new HashMap<>(
                        Map.of(
                            CustomServiceSettings.JSON_PARSER,
                            new HashMap<>(
                                Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                            )
                        )
                    )
                )
            ),
            ConfigurationParseContext.REQUEST,
            TaskType.TEXT_EMBEDDING
        );

        MatcherAssert.assertThat(
            settings,
            is(
                new CustomServiceSettings(
                    new CustomServiceSettings.TextEmbeddingSettings(SimilarityMeasure.DOT_PRODUCT, dims, maxInputTokens),
                    url,
                    Map.of("value", "abc"),
                    null,
                    requestContentString,
                    responseParser,
                    new RateLimitSettings(10_000)
                )
            )
        );
    }

    public void testFromMap_ReturnsError_IfHeadersContainsNonStringValues() {
        String similarity = SimilarityMeasure.DOT_PRODUCT.toString();
        Integer dims = 1536;
        Integer maxInputTokens = 512;
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var mapSettings = new HashMap<String, Object>(
            Map.of(
                ServiceFields.SIMILARITY,
                similarity,
                ServiceFields.DIMENSIONS,
                dims,
                ServiceFields.MAX_INPUT_TOKENS,
                maxInputTokens,
                CustomServiceSettings.URL,
                url,
                CustomServiceSettings.HEADERS,
                new HashMap<>(Map.of("key", 1)),
                CustomServiceSettings.REQUEST,
                requestContentString,
                CustomServiceSettings.RESPONSE,
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.JSON_PARSER,
                        new HashMap<>(
                            Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                        )
                    )
                )
            )
        );

        var exception = expectThrows(
            ValidationException.class,
            () -> CustomServiceSettings.fromMap(mapSettings, ConfigurationParseContext.REQUEST, TaskType.TEXT_EMBEDDING)
        );

        assertThat(
            exception.getMessage(),
            is(
                "Validation Failed: 1: Map field [headers] has an entry that is not valid, [key => 1]. "
                    + "Value type of [1] is not one of [String].;"
            )
        );
    }

    public void testFromMap_ReturnsError_IfQueryParamsContainsNonStringValues() {
        String similarity = SimilarityMeasure.DOT_PRODUCT.toString();
        Integer dims = 1536;
        Integer maxInputTokens = 512;
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var mapSettings = new HashMap<>(
            Map.of(
                ServiceFields.SIMILARITY,
                similarity,
                ServiceFields.DIMENSIONS,
                dims,
                ServiceFields.MAX_INPUT_TOKENS,
                maxInputTokens,
                CustomServiceSettings.URL,
                url,
                QueryParameters.QUERY_PARAMETERS,
                List.of(List.of("key", 1)),
                CustomServiceSettings.REQUEST,
                requestContentString,
                CustomServiceSettings.RESPONSE,
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.JSON_PARSER,
                        new HashMap<>(
                            Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                        )
                    )
                )
            )
        );

        var exception = expectThrows(
            ValidationException.class,
            () -> CustomServiceSettings.fromMap(mapSettings, ConfigurationParseContext.REQUEST, TaskType.TEXT_EMBEDDING)
        );

        assertThat(
            exception.getMessage(),
            is(
                "Validation Failed: 1: [service_settings] failed to parse tuple list entry [0] "
                    + "for setting [query_parameters], the second element must be a string but was [Integer];"
            )
        );
    }

    public void testFromMap_ReturnsError_IfRequestMapIsMissing() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var mapSettings = new HashMap<String, Object>(
            Map.of(
                CustomServiceSettings.URL,
                url,
                CustomServiceSettings.HEADERS,
                new HashMap<>(Map.of("key", "value")),
                "invalid_request",
                requestContentString,
                CustomServiceSettings.RESPONSE,
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.JSON_PARSER,
                        new HashMap<>(
                            Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                        )
                    )
                )
            )
        );

        var exception = expectThrows(
            ValidationException.class,
            () -> CustomServiceSettings.fromMap(mapSettings, ConfigurationParseContext.REQUEST, TaskType.TEXT_EMBEDDING)
        );

        assertThat(exception.getMessage(), is("Validation Failed: 1: [service_settings] does not contain the required setting [request];"));
    }

    public void testFromMap_ReturnsError_IfResponseMapIsMissing() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var mapSettings = new HashMap<String, Object>(
            Map.of(
                CustomServiceSettings.URL,
                url,
                CustomServiceSettings.HEADERS,
                new HashMap<>(Map.of("key", "value")),
                CustomServiceSettings.REQUEST,
                requestContentString,
                "invalid_response",
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.JSON_PARSER,
                        new HashMap<>(
                            Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                        )
                    )
                )
            )
        );

        var exception = expectThrows(
            ValidationException.class,
            () -> CustomServiceSettings.fromMap(mapSettings, ConfigurationParseContext.REQUEST, TaskType.TEXT_EMBEDDING)
        );

        assertThat(
            exception.getMessage(),
            is(
                "Validation Failed: 1: [service_settings] does not contain the required setting [response];"
                    + "2: [service_settings.response] does not contain the required setting [json_parser];"
            )
        );
    }

    public void testFromMap_ReturnsError_IfJsonParserMapIsNotEmptyAfterParsing() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var mapSettings = new HashMap<String, Object>(
            Map.of(
                CustomServiceSettings.URL,
                url,
                CustomServiceSettings.HEADERS,
                new HashMap<>(Map.of("key", "value")),
                CustomServiceSettings.REQUEST,
                requestContentString,
                CustomServiceSettings.RESPONSE,
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.JSON_PARSER,
                        new HashMap<>(
                            Map.of(
                                TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS,
                                "$.result.embeddings[*].embedding",
                                "key",
                                "value"
                            )
                        )
                    )
                )
            )
        );

        var exception = expectThrows(
            ElasticsearchStatusException.class,
            () -> CustomServiceSettings.fromMap(mapSettings, ConfigurationParseContext.REQUEST, TaskType.TEXT_EMBEDDING)
        );

        assertThat(
            exception.getMessage(),
            is(
                "Configuration contains unknown settings [{key=value}] while parsing field [json_parser]"
                    + " for settings [custom_service_settings]"
            )
        );
    }

    public void testFromMap_ReturnsError_IfResponseMapIsNotEmptyAfterParsing() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var mapSettings = new HashMap<String, Object>(
            Map.of(
                CustomServiceSettings.URL,
                url,
                CustomServiceSettings.HEADERS,
                new HashMap<>(Map.of("key", "value")),
                CustomServiceSettings.REQUEST,
                requestContentString,
                CustomServiceSettings.RESPONSE,
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.JSON_PARSER,
                        new HashMap<>(
                            Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                        ),
                        "key",
                        "value"
                    )
                )
            )
        );

        var exception = expectThrows(
            ElasticsearchStatusException.class,
            () -> CustomServiceSettings.fromMap(mapSettings, ConfigurationParseContext.REQUEST, TaskType.TEXT_EMBEDDING)
        );

        assertThat(
            exception.getMessage(),
            is(
                "Configuration contains unknown settings [{key=value}] while parsing field [response]"
                    + " for settings [custom_service_settings]"
            )
        );
    }

    public void testFromMap_ReturnsError_IfTaskTypeIsInvalid() {
        String url = "http://www.abc.com";
        String requestContentString = "request body";

        var mapSettings = new HashMap<String, Object>(
            Map.of(
                CustomServiceSettings.URL,
                url,
                CustomServiceSettings.HEADERS,
                new HashMap<>(Map.of("key", "value")),
                CustomServiceSettings.REQUEST,
                requestContentString,
                CustomServiceSettings.RESPONSE,
                new HashMap<>(
                    Map.of(
                        CustomServiceSettings.JSON_PARSER,
                        new HashMap<>(
                            Map.of(TextEmbeddingResponseParser.TEXT_EMBEDDING_PARSER_EMBEDDINGS, "$.result.embeddings[*].embedding")
                        )
                    )
                )
            )
        );

        var exception = expectThrows(
            IllegalArgumentException.class,
            () -> CustomServiceSettings.fromMap(mapSettings, ConfigurationParseContext.REQUEST, TaskType.CHAT_COMPLETION)
        );

        assertThat(exception.getMessage(), is("Invalid task type received [chat_completion] while constructing response parser"));
    }

    public void testXContent() throws IOException {
        var entity = new CustomServiceSettings(
            CustomServiceSettings.TextEmbeddingSettings.NON_TEXT_EMBEDDING_TASK_TYPE_SETTINGS,
            "http://www.abc.com",
            Map.of("key", "value"),
            null,
            "string",
            new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.FLOAT),
            null
        );

        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        entity.toXContent(builder, null);
        String xContentResult = Strings.toString(builder);

        var expected = XContentHelper.stripWhitespace("""
            {
                "url": "http://www.abc.com",
                "headers": {
                    "key": "value"
                },
                "request": "string",
                "response": {
                    "json_parser": {
                        "text_embeddings": "$.result.embeddings[*].embedding",
                        "embedding_type": "float"
                    }
                },
                "input_type": {
                    "translation": {},
                    "default": ""
                },
                "rate_limit": {
                    "requests_per_minute": 10000
                },
                "batch_size": 10
            }
            """);

        assertThat(xContentResult, is(expected));
    }

    public void testXContent_Rerank() throws IOException {
        var entity = new CustomServiceSettings(
            CustomServiceSettings.TextEmbeddingSettings.NON_TEXT_EMBEDDING_TASK_TYPE_SETTINGS,
            "http://www.abc.com",
            Map.of("key", "value"),
            null,
            "string",
            new RerankResponseParser(
                "$.result.reranked_results[*].relevance_score",
                "$.result.reranked_results[*].index",
                "$.result.reranked_results[*].document_text"
            ),
            null
        );

        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        entity.toXContent(builder, null);
        String xContentResult = Strings.toString(builder);

        var expected = XContentHelper.stripWhitespace("""
            {
                "url": "http://www.abc.com",
                "headers": {
                    "key": "value"
                },
                "request": "string",
                "response": {
                    "json_parser": {
                        "relevance_score": "$.result.reranked_results[*].relevance_score",
                        "reranked_index": "$.result.reranked_results[*].index",
                        "document_text": "$.result.reranked_results[*].document_text"
                    }
                },
                "input_type": {
                    "translation": {},
                    "default": ""
                },
                "rate_limit": {
                    "requests_per_minute": 10000
                },
                "batch_size": 10
            }
            """);

        assertThat(xContentResult, is(expected));
    }

    public void testXContent_WithInputTypeTranslationValues() throws IOException {
        var entity = new CustomServiceSettings(
            CustomServiceSettings.TextEmbeddingSettings.NON_TEXT_EMBEDDING_TASK_TYPE_SETTINGS,
            "http://www.abc.com",
            Map.of("key", "value"),
            null,
            "string",
            new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.FLOAT),
            null,
            null,
            new InputTypeTranslator(Map.of(InputType.SEARCH, "do_search", InputType.INGEST, "do_ingest"), "a_default")
        );

        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        entity.toXContent(builder, null);
        String xContentResult = Strings.toString(builder);

        var expected = XContentHelper.stripWhitespace("""
            {
                "url": "http://www.abc.com",
                "headers": {
                    "key": "value"
                },
                "request": "string",
                "response": {
                    "json_parser": {
                        "text_embeddings": "$.result.embeddings[*].embedding",
                        "embedding_type": "float"
                    }
                },
                "input_type": {
                    "translation": {
                        "ingest": "do_ingest",
                        "search": "do_search"
                    },
                    "default": "a_default"
                },
                "rate_limit": {
                    "requests_per_minute": 10000
                },
                "batch_size": 10
            }
            """);

        assertThat(xContentResult, is(expected));
    }

    public void testXContent_BatchSize11() throws IOException {
        var entity = new CustomServiceSettings(
            CustomServiceSettings.TextEmbeddingSettings.NON_TEXT_EMBEDDING_TASK_TYPE_SETTINGS,
            "http://www.abc.com",
            Map.of("key", "value"),
            null,
            "string",
            new TextEmbeddingResponseParser("$.result.embeddings[*].embedding", CustomServiceEmbeddingType.FLOAT),
            null,
            11,
            InputTypeTranslator.EMPTY_TRANSLATOR
        );

        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        entity.toXContent(builder, null);
        String xContentResult = Strings.toString(builder);

        var expected = XContentHelper.stripWhitespace("""
            {
                "url": "http://www.abc.com",
                "headers": {
                    "key": "value"
                },
                "request": "string",
                "response": {
                    "json_parser": {
                        "text_embeddings": "$.result.embeddings[*].embedding",
                        "embedding_type": "float"
                    }
                },
                "input_type": {
                    "translation": {},
                    "default": ""
                },
                "rate_limit": {
                    "requests_per_minute": 10000
                },
                "batch_size": 11
            }
            """);

        assertThat(xContentResult, is(expected));
    }

    @Override
    protected NamedWriteableRegistry getNamedWriteableRegistry() {
        return new NamedWriteableRegistry(InferenceNamedWriteablesProvider.getNamedWriteables());
    }

    @Override
    protected Writeable.Reader<CustomServiceSettings> instanceReader() {
        return CustomServiceSettings::new;
    }

    @Override
    protected CustomServiceSettings createTestInstance() {
        return createRandom(randomAlphaOfLength(5));
    }

    @Override
    protected CustomServiceSettings mutateInstance(CustomServiceSettings instance) {
        return randomValueOtherThan(instance, CustomServiceSettingsTests::createRandom);
    }

    @Override
    protected CustomServiceSettings mutateInstanceForVersion(CustomServiceSettings instance, TransportVersion version) {
        return instance;
    }
}
