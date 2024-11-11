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
package com.facebook.presto.sql.planner.plan;

import com.facebook.presto.spi.plan.Assignments;
import com.facebook.presto.spi.plan.PlanNode;
import com.facebook.presto.spi.plan.PlanNodeId;
import com.facebook.presto.spi.plan.ProjectNode;
import com.facebook.presto.spi.relation.DeterminismEvaluator;
import com.facebook.presto.spi.relation.RowExpression;
import com.facebook.presto.spi.relation.VariableReferenceExpression;
import com.facebook.presto.sql.planner.CanonicalJoinNode;
import com.facebook.presto.sql.planner.iterative.Lookup;
import com.facebook.presto.sql.relational.FunctionResolution;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static com.facebook.presto.expressions.LogicalRowExpressions.TRUE_CONSTANT;
import static com.facebook.presto.expressions.LogicalRowExpressions.and;
import static com.facebook.presto.expressions.LogicalRowExpressions.extractConjuncts;
import static com.facebook.presto.expressions.RowExpressionNodeInliner.replaceExpression;
import static com.facebook.presto.spi.plan.JoinType.INNER;
import static com.facebook.presto.sql.planner.VariablesExtractor.extractUnique;
import static com.facebook.presto.sql.planner.optimizations.JoinNodeUtils.toRowExpression;
import static com.facebook.presto.sql.planner.plan.AssignmentUtils.getNonIdentityAssignments;
import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.Objects.requireNonNull;

public class MultiJoinNode
{
    // Use a linked hash set to ensure optimizer is deterministic
    private final CanonicalJoinNode node;
    private final Assignments assignments;

    public MultiJoinNode(LinkedHashSet<PlanNode> sources, RowExpression filter, List<VariableReferenceExpression> outputVariables,
            Assignments assignments)
    {
        checkArgument(sources.size() > 1, "sources size is <= 1");

        requireNonNull(sources, "sources is null");
        requireNonNull(filter, "filter is null");
        requireNonNull(outputVariables, "outputVariables is null");
        requireNonNull(assignments, "assignments is null");

        this.assignments = assignments;
        // Plan node id doesn't matter here as we don't use this in planner
        this.node = new CanonicalJoinNode(
                new PlanNodeId(""),
                sources.stream().collect(toImmutableList()),
                INNER,
                ImmutableSet.of(),
                ImmutableSet.of(filter),
                outputVariables);
    }

    public RowExpression getFilter()
    {
        return node.getFilters().stream().findAny().get();
    }

    public LinkedHashSet<PlanNode> getSources()
    {
        return new LinkedHashSet<>(node.getSources());
    }

    public List<VariableReferenceExpression> getOutputVariables()
    {
        return node.getOutputVariables();
    }

    public Assignments getAssignments()
    {
        return assignments;
    }

    public static Builder builder()
    {
        return new Builder();
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(getSources(), ImmutableSet.copyOf(extractConjuncts(getFilter())), getOutputVariables());
    }

    @Override
    public boolean equals(Object obj)
    {
        if (!(obj instanceof MultiJoinNode)) {
            return false;
        }

        MultiJoinNode other = (MultiJoinNode) obj;
        return getSources().equals(other.getSources())
                && ImmutableSet.copyOf(extractConjuncts(getFilter())).equals(ImmutableSet.copyOf(extractConjuncts(other.getFilter())))
                && getOutputVariables().equals(other.getOutputVariables())
                && getAssignments().equals(other.getAssignments());
    }

    @Override
    public String toString()
    {
        return "MultiJoinNode{" +
                "node=" + node +
                ", assignments=" + assignments +
                '}';
    }

    static MultiJoinNode toMultiJoinNode(JoinNode joinNode, Lookup lookup, int joinLimit, boolean handleComplexEquiJoins, FunctionResolution functionResolution, DeterminismEvaluator determinismEvaluator)
    {
        // the number of sources is the number of joins + 1
        return new MultiJoinNode.JoinNodeFlattener(joinNode, lookup, joinLimit + 1, handleComplexEquiJoins, functionResolution, determinismEvaluator).toMultiJoinNode();
    }

    private static class JoinNodeFlattener
    {
        private final LinkedHashSet<PlanNode> sources = new LinkedHashSet<>();
        private final Assignments intermediateAssignments;
        private final boolean handleComplexEquiJoins;
        private List<RowExpression> filters = new ArrayList<>();
        private final List<VariableReferenceExpression> outputVariables;
        private final FunctionResolution functionResolution;
        private final DeterminismEvaluator determinismEvaluator;
        private final Lookup lookup;

        JoinNodeFlattener(JoinNode node, Lookup lookup, int sourceLimit, boolean handleComplexEquiJoins, FunctionResolution functionResolution,
                DeterminismEvaluator determinismEvaluator)
        {
            requireNonNull(node, "node is null");
            checkState(node.getType() == INNER, "join type must be INNER");
            this.outputVariables = node.getOutputVariables();
            this.lookup = requireNonNull(lookup, "lookup is null");
            this.functionResolution = requireNonNull(functionResolution, "functionResolution is null");
            this.determinismEvaluator = requireNonNull(determinismEvaluator, "determinismEvaluator is null");
            this.handleComplexEquiJoins = handleComplexEquiJoins;

            Map<VariableReferenceExpression, RowExpression> intermediateAssignments = new HashMap<>();
            flattenNode(node, sourceLimit, intermediateAssignments);

            // We resolve the intermediate assignments to only inputs of the flattened join node
            ImmutableSet<VariableReferenceExpression> inputVariables = sources.stream().flatMap(s -> s.getOutputVariables().stream()).collect(toImmutableSet());
            this.intermediateAssignments = resolveAssignments(intermediateAssignments, inputVariables);
            rewriteFilterWithInlinedAssignments(this.intermediateAssignments);
        }

        private Assignments resolveAssignments(Map<VariableReferenceExpression, RowExpression> assignments, Set<VariableReferenceExpression> availableVariables)
        {
            HashSet<VariableReferenceExpression> resolvedVariables = new HashSet<>();
            ImmutableList.copyOf(assignments.keySet()).forEach(variable -> resolveVariable(variable, resolvedVariables, assignments, availableVariables));

            return Assignments.builder().putAll(assignments).build();
        }

        private void resolveVariable(VariableReferenceExpression variable, HashSet<VariableReferenceExpression> resolvedVariables, Map<VariableReferenceExpression,
                RowExpression> assignments, Set<VariableReferenceExpression> availableVariables)
        {
            RowExpression expression = assignments.get(variable);
            Sets.SetView<VariableReferenceExpression> variablesToResolve = Sets.difference(Sets.difference(extractUnique(expression), availableVariables), resolvedVariables);

            // Recursively resolve any unresolved variables
            variablesToResolve.forEach(variableToResolve -> resolveVariable(variableToResolve, resolvedVariables, assignments, availableVariables));

            // Modify the assignment for the variable : Replace it with the now resolved constituent variables
            assignments.put(variable, replaceExpression(expression, assignments));
            // Mark this variable as resolved
            resolvedVariables.add(variable);
        }

        private void rewriteFilterWithInlinedAssignments(Assignments assignments)
        {
            ImmutableList.Builder<RowExpression> modifiedFilters = ImmutableList.builder();
            filters.forEach(filter -> modifiedFilters.add(replaceExpression(filter, assignments.getMap())));
            filters = modifiedFilters.build();
        }

        private void flattenNode(PlanNode node, int limit, Map<VariableReferenceExpression, RowExpression> assignmentsBuilder)
        {
            PlanNode resolved = lookup.resolve(node);

            if (resolved instanceof ProjectNode) {
                ProjectNode projectNode = (ProjectNode) resolved;
                // A ProjectNode could be 'hiding' a join source by building an assignment of a complex equi-join criteria like `left.key = right1.key1 + right1.key2`
                // We open up the join space by tracking the assignments from this Project node; these will be inlined into the overall filters once we finish
                // traversing the join graph
                // We only do this if the ProjectNode assignments are deterministic
                if (handleComplexEquiJoins && lookup.resolve(projectNode.getSource()) instanceof JoinNode &&
                        projectNode.getAssignments().getExpressions().stream().allMatch(determinismEvaluator::isDeterministic)) {
                    //We keep track of only the non-identity assignments since these are the ones that will be inlined into the overall filters
                    assignmentsBuilder.putAll(getNonIdentityAssignments(projectNode.getAssignments()));
                    flattenNode(projectNode.getSource(), limit, assignmentsBuilder);
                }
                else {
                    sources.add(node);
                }
                return;
            }

            // (limit - 2) because you need to account for adding left and right side
            if (!(resolved instanceof JoinNode) || (sources.size() > (limit - 2))) {
                sources.add(node);
                return;
            }

            JoinNode joinNode = (JoinNode) resolved;
            if (joinNode.getType() != INNER || !determinismEvaluator.isDeterministic(joinNode.getFilter().orElse(TRUE_CONSTANT)) || joinNode.getDistributionType().isPresent()) {
                sources.add(node);
                return;
            }

            // we set the left limit to limit - 1 to account for the node on the right
            flattenNode(joinNode.getLeft(), limit - 1, assignmentsBuilder);
            flattenNode(joinNode.getRight(), limit, assignmentsBuilder);
            joinNode.getCriteria().stream()
                    .map(criteria -> toRowExpression(criteria, functionResolution))
                    .forEach(filters::add);
            joinNode.getFilter().ifPresent(filters::add);
        }

        MultiJoinNode toMultiJoinNode()
        {
            ImmutableSet<VariableReferenceExpression> inputVariables = sources.stream().flatMap(source -> source.getOutputVariables().stream()).collect(toImmutableSet());

            // We could have some output variables that were possibly generated from intermediate assignments
            // For each of these variables, use the intermediate assignments to replace this variable with the set of input variables it uses

            // Additionally, we build an overall set of assignments for the reordered Join node - this is used to add a wrapper Project over the updated output variables
            // We do this to satisfy the invariant that the rewritten Join node must produce the same output variables as the input Join node
            ImmutableSet.Builder<VariableReferenceExpression> updatedOutputVariables = ImmutableSet.builder();
            Assignments.Builder overallAssignments = Assignments.builder();
            boolean nonIdentityAssignmentsFound = false;

            for (VariableReferenceExpression outputVariable : outputVariables) {
                if (inputVariables.contains(outputVariable)) {
                    overallAssignments.put(outputVariable, outputVariable);
                    updatedOutputVariables.add(outputVariable);
                    continue;
                }

                checkState(intermediateAssignments.getMap().containsKey(outputVariable),
                        "Output variable [%s] not found in input variables or in intermediate assignments", outputVariable);
                nonIdentityAssignmentsFound = true;
                overallAssignments.put(outputVariable, intermediateAssignments.get(outputVariable));
                updatedOutputVariables.addAll(extractUnique(intermediateAssignments.get(outputVariable)));
            }

            return new MultiJoinNode(sources,
                    and(filters),
                    updatedOutputVariables.build().asList(),
                    nonIdentityAssignmentsFound ? overallAssignments.build() : Assignments.of());
        }
    }

    static class Builder
    {
        private List<PlanNode> sources;
        private RowExpression filter;
        private List<VariableReferenceExpression> outputVariables;
        private Assignments assignments = Assignments.of();

        public MultiJoinNode.Builder setSources(PlanNode... sources)
        {
            this.sources = ImmutableList.copyOf(sources);
            return this;
        }

        public MultiJoinNode.Builder setFilter(RowExpression filter)
        {
            this.filter = filter;
            return this;
        }

        public MultiJoinNode.Builder setAssignments(Assignments assignments)
        {
            this.assignments = assignments;
            return this;
        }

        public Builder setOutputVariables(VariableReferenceExpression... outputVariables)
        {
            this.outputVariables = ImmutableList.copyOf(outputVariables);
            return this;
        }

        public MultiJoinNode build()
        {
            return new MultiJoinNode(new LinkedHashSet<>(sources), filter, outputVariables, assignments);
        }
    }
}
