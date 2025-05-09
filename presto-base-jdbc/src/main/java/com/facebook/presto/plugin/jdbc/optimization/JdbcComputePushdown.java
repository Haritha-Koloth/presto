/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.facebook.presto.plugin.jdbc.optimization;

import com.facebook.presto.expressions.LogicalRowExpressions;
import com.facebook.presto.expressions.translator.TranslatedExpression;
import com.facebook.presto.plugin.jdbc.JdbcTableHandle;
import com.facebook.presto.plugin.jdbc.JdbcTableLayoutHandle;
import com.facebook.presto.spi.ConnectorPlanOptimizer;
import com.facebook.presto.spi.ConnectorPlanRewriter;
import com.facebook.presto.spi.ConnectorSession;
import com.facebook.presto.spi.TableHandle;
import com.facebook.presto.spi.VariableAllocator;
import com.facebook.presto.spi.function.FunctionMetadataManager;
import com.facebook.presto.spi.function.StandardFunctionResolution;
import com.facebook.presto.spi.plan.FilterNode;
import com.facebook.presto.spi.plan.PlanNode;
import com.facebook.presto.spi.plan.PlanNodeIdAllocator;
import com.facebook.presto.spi.plan.TableScanNode;
import com.facebook.presto.spi.relation.DeterminismEvaluator;
import com.facebook.presto.spi.relation.ExpressionOptimizerProvider;
import com.facebook.presto.spi.relation.RowExpression;

import java.util.Optional;
import java.util.Set;

import static com.facebook.presto.expressions.translator.FunctionTranslator.buildFunctionTranslator;
import static com.facebook.presto.expressions.translator.RowExpressionTreeTranslator.translateWith;
import static com.facebook.presto.spi.ConnectorPlanRewriter.rewriteWith;
import static com.facebook.presto.spi.relation.ExpressionOptimizer.Level.OPTIMIZED;
import static java.util.Objects.requireNonNull;

public class JdbcComputePushdown
        implements ConnectorPlanOptimizer
{
    private final ExpressionOptimizerProvider expressionOptimizerProvider;
    private final JdbcFilterToSqlTranslator jdbcFilterToSqlTranslator;
    private final LogicalRowExpressions logicalRowExpressions;

    public JdbcComputePushdown(
            FunctionMetadataManager functionMetadataManager,
            StandardFunctionResolution functionResolution,
            DeterminismEvaluator determinismEvaluator,
            ExpressionOptimizerProvider expressionOptimizerProvider,
            String identifierQuote,
            Set<Class<?>> functionTranslators)
    {
        requireNonNull(functionMetadataManager, "functionMetadataManager is null");
        requireNonNull(identifierQuote, "identifierQuote is null");
        requireNonNull(functionTranslators, "functionTranslators is null");
        requireNonNull(determinismEvaluator, "determinismEvaluator is null");
        requireNonNull(functionResolution, "functionResolution is null");

        this.expressionOptimizerProvider = requireNonNull(expressionOptimizerProvider, "expressionOptimizerProvider is null");
        this.jdbcFilterToSqlTranslator = new JdbcFilterToSqlTranslator(
                functionMetadataManager,
                buildFunctionTranslator(functionTranslators),
                identifierQuote);
        this.logicalRowExpressions = new LogicalRowExpressions(
                determinismEvaluator,
                functionResolution,
                functionMetadataManager);
    }

    @Override
    public PlanNode optimize(
            PlanNode maxSubplan,
            ConnectorSession session,
            VariableAllocator variableAllocator,
            PlanNodeIdAllocator idAllocator)
    {
        return rewriteWith(new Rewriter(session, idAllocator), maxSubplan);
    }

    private class Rewriter
            extends ConnectorPlanRewriter<Void>
    {
        private final ConnectorSession session;
        private final PlanNodeIdAllocator idAllocator;

        public Rewriter(ConnectorSession session, PlanNodeIdAllocator idAllocator)
        {
            this.session = requireNonNull(session, "session is null");
            this.idAllocator = requireNonNull(idAllocator, "idAllocator is null");
        }

        @Override
        public PlanNode visitFilter(FilterNode node, RewriteContext<Void> context)
        {
            if (!(node.getSource() instanceof TableScanNode)) {
                return node;
            }

            TableScanNode oldTableScanNode = (TableScanNode) node.getSource();
            TableHandle oldTableHandle = oldTableScanNode.getTable();
            JdbcTableHandle oldConnectorTable = (JdbcTableHandle) oldTableHandle.getConnectorHandle();

            RowExpression predicate = expressionOptimizerProvider.getExpressionOptimizer(session).optimize(node.getPredicate(), OPTIMIZED, session);
            predicate = logicalRowExpressions.convertToConjunctiveNormalForm(predicate);
            TranslatedExpression<JdbcExpression> jdbcExpression = translateWith(
                    predicate,
                    jdbcFilterToSqlTranslator,
                    oldTableScanNode.getAssignments());

            // TODO if jdbcExpression is not present, walk through translated subtree to find out which parts can be pushed down
            if (!oldTableHandle.getLayout().isPresent() || !jdbcExpression.getTranslated().isPresent()) {
                return node;
            }

            JdbcTableLayoutHandle oldTableLayoutHandle = (JdbcTableLayoutHandle) oldTableHandle.getLayout().get();
            JdbcTableLayoutHandle newTableLayoutHandle = new JdbcTableLayoutHandle(
                    session.getSqlFunctionProperties(),
                    oldConnectorTable,
                    oldTableLayoutHandle.getTupleDomain(),
                    jdbcExpression.getTranslated());

            TableHandle tableHandle = new TableHandle(
                    oldTableHandle.getConnectorId(),
                    oldTableHandle.getConnectorHandle(),
                    oldTableHandle.getTransaction(),
                    Optional.of(newTableLayoutHandle));

            TableScanNode newTableScanNode = new TableScanNode(
                    oldTableScanNode.getSourceLocation(),
                    idAllocator.getNextId(),
                    tableHandle,
                    oldTableScanNode.getOutputVariables(),
                    oldTableScanNode.getAssignments(),
                    oldTableScanNode.getTableConstraints(),
                    oldTableScanNode.getCurrentConstraint(),
                    oldTableScanNode.getEnforcedConstraint(),
                    oldTableScanNode.getCteMaterializationInfo());

            return new FilterNode(node.getSourceLocation(), idAllocator.getNextId(), newTableScanNode, node.getPredicate());
        }
    }
}
