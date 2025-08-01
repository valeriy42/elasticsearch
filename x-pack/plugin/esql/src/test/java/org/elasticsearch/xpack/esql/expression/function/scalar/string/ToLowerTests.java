/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.expression.function.scalar.string;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.apache.lucene.util.BytesRef;
import org.elasticsearch.common.lucene.BytesRefs;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.compute.data.Block;
import org.elasticsearch.compute.data.Page;
import org.elasticsearch.xpack.esql.EsqlTestUtils;
import org.elasticsearch.xpack.esql.core.expression.Expression;
import org.elasticsearch.xpack.esql.core.expression.FoldContext;
import org.elasticsearch.xpack.esql.core.expression.Literal;
import org.elasticsearch.xpack.esql.core.tree.Source;
import org.elasticsearch.xpack.esql.core.type.DataType;
import org.elasticsearch.xpack.esql.core.util.DateUtils;
import org.elasticsearch.xpack.esql.expression.function.TestCaseSupplier;
import org.elasticsearch.xpack.esql.expression.function.scalar.AbstractConfigurationFunctionTestCase;
import org.elasticsearch.xpack.esql.plugin.EsqlPlugin;
import org.elasticsearch.xpack.esql.plugin.QueryPragmas;
import org.elasticsearch.xpack.esql.session.Configuration;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.hamcrest.Matchers.equalTo;

public class ToLowerTests extends AbstractConfigurationFunctionTestCase {
    public ToLowerTests(@Name("TestCase") Supplier<TestCaseSupplier.TestCase> testCaseSupplier) {
        this.testCase = testCaseSupplier.get();
    }

    @ParametersFactory
    public static Iterable<Object[]> parameters() {
        List<TestCaseSupplier> suppliers = new ArrayList<>();

        suppliers(suppliers, "keyword ascii", DataType.KEYWORD, () -> randomAlphaOfLengthBetween(1, 10));
        suppliers(suppliers, "keyword unicode", DataType.KEYWORD, () -> randomUnicodeOfLengthBetween(1, 10));
        suppliers(suppliers, "text ascii", DataType.TEXT, () -> randomAlphaOfLengthBetween(1, 10));
        suppliers(suppliers, "text unicode", DataType.TEXT, () -> randomUnicodeOfLengthBetween(1, 10));
        return parameterSuppliersFromTypedDataWithDefaultChecksNoErrors(true, suppliers);
    }

    public void testRandomLocale() {
        String testString = randomAlphaOfLength(10);
        Configuration cfg = randomLocaleConfig();
        ToLower func = new ToLower(Source.EMPTY, Literal.keyword(Source.EMPTY, testString), cfg);
        assertThat(BytesRefs.toBytesRef(testString.toLowerCase(cfg.locale())), equalTo(func.fold(FoldContext.small())));
    }

    private Configuration randomLocaleConfig() {
        return new Configuration(
            DateUtils.UTC,
            randomLocale(random()),
            null,
            null,
            new QueryPragmas(Settings.EMPTY),
            EsqlPlugin.QUERY_RESULT_TRUNCATION_MAX_SIZE.getDefault(Settings.EMPTY),
            EsqlPlugin.QUERY_RESULT_TRUNCATION_DEFAULT_SIZE.getDefault(Settings.EMPTY),
            "",
            false,
            Map.of(),
            System.nanoTime(),
            randomBoolean()
        );
    }

    @Override
    protected Expression buildWithConfiguration(Source source, List<Expression> args, Configuration configuration) {
        return new ToLower(source, args.get(0), configuration);
    }

    private static void suppliers(List<TestCaseSupplier> suppliers, String name, DataType type, Supplier<String> valueSupplier) {
        suppliers.add(new TestCaseSupplier(name, List.of(type), () -> {
            List<TestCaseSupplier.TypedData> values = new ArrayList<>();
            String expectedToString = "ChangeCaseEvaluator[val=Attribute[channel=0], locale=en_US, caseType=LOWER]";

            String value = valueSupplier.get();
            values.add(new TestCaseSupplier.TypedData(new BytesRef(value), type, "0"));

            String expectedValue = value.toLowerCase(EsqlTestUtils.TEST_CFG.locale());
            return new TestCaseSupplier.TestCase(values, expectedToString, type, equalTo(new BytesRef(expectedValue)));
        }));
        suppliers.add(new TestCaseSupplier(name + " mv", List.of(type), () -> {
            List<TestCaseSupplier.TypedData> values = new ArrayList<>();
            String expectedToString = "ChangeCaseEvaluator[val=Attribute[channel=0], locale=en_US, caseType=LOWER]";

            List<String> strings = randomList(2, 10, valueSupplier);
            values.add(new TestCaseSupplier.TypedData(strings.stream().map(BytesRef::new).toList(), type, "0"));

            List<BytesRef> expectedValue = strings.stream().map(s -> new BytesRef(s.toLowerCase(EsqlTestUtils.TEST_CFG.locale()))).toList();
            return new TestCaseSupplier.TestCase(values, expectedToString, type, equalTo(expectedValue));
        }));
    }

    @Override
    protected void extraBlockTests(Page in, Block out) {
        assertIsOrdIfInIsOrd(in, out);
    }
}
