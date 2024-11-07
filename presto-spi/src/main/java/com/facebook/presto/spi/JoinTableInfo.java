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
package com.facebook.presto.spi;

import com.facebook.presto.spi.relation.VariableReferenceExpression;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import java.util.List;
import java.util.Map;

public class JoinTableInfo
{
    private final ConnectorTableHandle tableHandle;
    private final Map<VariableReferenceExpression, ColumnHandle> assignments;
    private final List<VariableReferenceExpression> outputVariables;

    public JoinTableInfo(ConnectorTableHandle tableHandle, Map<VariableReferenceExpression, ColumnHandle> assignments, List<VariableReferenceExpression> outputVariables)
    {
        this.tableHandle = tableHandle;
        this.assignments = ImmutableMap.copyOf(assignments);
        this.outputVariables = ImmutableList.copyOf(outputVariables);
    }

    public ConnectorTableHandle getTableHandle()
    {
        return tableHandle;
    }

    public Map<VariableReferenceExpression, ColumnHandle> getAssignments()
    {
        return assignments;
    }

    public List<VariableReferenceExpression> getOutputVariables()
    {
        return outputVariables;
    }
}
