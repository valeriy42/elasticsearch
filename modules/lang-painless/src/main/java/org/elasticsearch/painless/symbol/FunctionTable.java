/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.painless.symbol;

import org.elasticsearch.painless.lookup.PainlessLookupUtility;
import org.objectweb.asm.commons.Method;

import java.lang.invoke.MethodType;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Stores information about functions directly invokable on the generated script class.
 * Functions marked as internal are generated by lambdas or method references, and are
 * not directly callable by a user.
 */
public class FunctionTable {

    private static final String MANGLED_FUNCTION_NAME_PREFIX = "&";

    public static class LocalFunction {

        protected final String functionName;
        protected final String mangledName;
        protected final Class<?> returnType;
        protected final List<Class<?>> typeParameters;
        protected final boolean isInternal;
        protected final boolean isStatic;

        protected final MethodType methodType;
        protected final Method asmMethod;

        public LocalFunction(
            String functionName,
            Class<?> returnType,
            List<Class<?>> typeParameters,
            boolean isInternal,
            boolean isStatic
        ) {
            this(functionName, "", returnType, typeParameters, isInternal, isStatic);
        }

        private LocalFunction(
            String functionName,
            String mangle,
            Class<?> returnType,
            List<Class<?>> typeParameters,
            boolean isInternal,
            boolean isStatic
        ) {

            this.functionName = Objects.requireNonNull(functionName);
            this.mangledName = Objects.requireNonNull(mangle) + this.functionName;
            this.returnType = Objects.requireNonNull(returnType);
            this.typeParameters = List.copyOf(typeParameters);
            this.isInternal = isInternal;
            this.isStatic = isStatic;

            Class<?> javaReturnType = PainlessLookupUtility.typeToJavaType(returnType);
            Class<?>[] javaTypeParameters = typeParameters.stream().map(PainlessLookupUtility::typeToJavaType).toArray(Class<?>[]::new);

            this.methodType = MethodType.methodType(javaReturnType, javaTypeParameters);
            this.asmMethod = new org.objectweb.asm.commons.Method(
                mangledName,
                MethodType.methodType(javaReturnType, javaTypeParameters).toMethodDescriptorString()
            );
        }

        public String getMangledName() {
            return mangledName;
        }

        public Class<?> getReturnType() {
            return returnType;
        }

        public List<Class<?>> getTypeParameters() {
            return typeParameters;
        }

        public boolean isInternal() {
            return isInternal;
        }

        public boolean isStatic() {
            return isStatic;
        }

        public MethodType getMethodType() {
            return methodType;
        }

        public Method getAsmMethod() {
            return asmMethod;
        }
    }

    /**
     * Generates a {@code LocalFunction} key.
     * @param functionName the name of the {@code LocalFunction}
     * @param functionArity the number of parameters for the {@code LocalFunction}
     * @return a {@code LocalFunction} key used for {@code LocalFunction} look up within the {@code FunctionTable}
     */
    public static String buildLocalFunctionKey(String functionName, int functionArity) {
        return functionName + "/" + functionArity;
    }

    protected Map<String, LocalFunction> localFunctions = new HashMap<>();

    public LocalFunction addFunction(
        String functionName,
        Class<?> returnType,
        List<Class<?>> typeParameters,
        boolean isInternal,
        boolean isStatic
    ) {

        String functionKey = buildLocalFunctionKey(functionName, typeParameters.size());
        LocalFunction function = new LocalFunction(functionName, returnType, typeParameters, isInternal, isStatic);
        localFunctions.put(functionKey, function);
        return function;
    }

    public LocalFunction addMangledFunction(
        String functionName,
        Class<?> returnType,
        List<Class<?>> typeParameters,
        boolean isInternal,
        boolean isStatic
    ) {
        String functionKey = buildLocalFunctionKey(functionName, typeParameters.size());
        LocalFunction function = new LocalFunction(
            functionName,
            MANGLED_FUNCTION_NAME_PREFIX,
            returnType,
            typeParameters,
            isInternal,
            isStatic
        );
        localFunctions.put(functionKey, function);
        return function;
    }

    public LocalFunction getFunction(String functionName, int functionArity) {
        String functionKey = buildLocalFunctionKey(functionName, functionArity);
        return localFunctions.get(functionKey);
    }

    public LocalFunction getFunction(String functionKey) {
        return localFunctions.get(functionKey);
    }
}
