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
package com.facebook.presto.connector.thrift.integration;

import com.facebook.presto.testing.QueryRunner;
import com.facebook.presto.tests.AbstractTestQueries;
import com.google.common.collect.ImmutableMap;
import org.testng.annotations.Test;

import static com.facebook.presto.connector.thrift.integration.ThriftQueryRunner.createThriftQueryRunner;

public class TestThriftDistributedQueries
        extends AbstractTestQueries
{
    @Override
    public void testAssignUniqueId()
    {
        // this test can take a long time
    }

    @Override
    protected QueryRunner createQueryRunner()
            throws Exception
    {
        return createThriftQueryRunner(3, 3, false, ImmutableMap.of());
    }

    @Test
    public void testQuotedIdentifiers()
    {
        // Expected to fail as Table is stored in Uppercase in H2 db and exists in tpch as lowercase
        assertQueryFails("SELECT \"TOTALPRICE\" \"my price\" FROM \"ORDERS\"", "Table thrift.tiny.ORDERS does not exist");
    }
}
