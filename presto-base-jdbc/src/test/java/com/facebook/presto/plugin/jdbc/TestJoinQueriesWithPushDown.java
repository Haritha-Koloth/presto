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
package com.facebook.presto.plugin.jdbc;

import com.facebook.presto.Session;
import com.facebook.presto.execution.QueryInfo;
import com.facebook.presto.spi.QueryId;
import com.facebook.presto.testing.QueryRunner;
import com.facebook.presto.tests.AbstractTestJoinQueries;
import com.facebook.presto.tests.DistributedQueryRunner;
import com.facebook.presto.tpch.TpchPlugin;
import com.google.common.collect.ImmutableList;
import io.airlift.tpch.TpchTable;
import org.testng.annotations.Test;

import java.util.Map;

import static com.facebook.airlift.testing.Closeables.closeAllSuppress;
import static com.facebook.presto.SystemSessionProperties.INNER_JOIN_PUSHDOWN_ENABLED;
import static com.facebook.presto.SystemSessionProperties.JOIN_DISTRIBUTION_TYPE;
import static com.facebook.presto.SystemSessionProperties.JOIN_REORDERING_STRATEGY;
import static com.facebook.presto.plugin.jdbc.JdbcQueryRunner.createSchema;
import static com.facebook.presto.sql.analyzer.FeaturesConfig.JoinDistributionType.BROADCAST;
import static com.facebook.presto.sql.analyzer.FeaturesConfig.JoinDistributionType.PARTITIONED;
import static com.facebook.presto.sql.analyzer.FeaturesConfig.JoinReorderingStrategy.NONE;
import static com.facebook.presto.testing.TestingSession.testSessionBuilder;
import static com.facebook.presto.tests.QueryAssertions.copyTpchTables;
import static com.facebook.presto.tpch.TpchMetadata.TINY_SCHEMA_NAME;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotEquals;

public class TestJoinQueriesWithPushDown
        extends AbstractTestJoinQueries
{
    private static final String TPCH_SCHEMA = "tpch";
    private static final String JDBC = "jdbc";
    @Override
    protected QueryRunner createQueryRunner() throws Exception
    {
        Session session = testSessionBuilder()
                .setCatalog(JDBC)
                .setSchema(TPCH_SCHEMA)
                .setSystemProperty("optimizer_inner_join_pushdown_enabled", "true")
                .setSystemProperty("optimizer_inequality_join_pushdown_enabled", "true").build();
        DistributedQueryRunner queryRunner = null;
        try {
            queryRunner = new DistributedQueryRunner(session, 3);

            queryRunner.installPlugin(new TpchPlugin());
            queryRunner.createCatalog("tpch", "tpch");

            Map<String, String> properties = TestingH2JdbcModule.createProperties();
            createSchema(properties, "tpch");

            queryRunner.installPlugin(new JdbcPlugin("base-jdbc", new TestingH2JdbcModule()));
            queryRunner.createCatalog(JDBC, "base-jdbc", properties);

            copyTpchTables(queryRunner, "tpch", TINY_SCHEMA_NAME, session, ImmutableList.copyOf(TpchTable.getTables()));

            return queryRunner;
        }
        catch (Throwable e) {
            closeAllSuppress(e, queryRunner);
            throw e;
        }
    }

    @Test
    public void testSimpleInnerEquiJoin()
    {
        Session pushdownEnabled = getSession(true);
        Session disabled = getSession(false);
        assertQueryWithSameQueryRunner(pushdownEnabled, "SELECT o.orderkey, o.totalprice, c.name " +
                "FROM orders o " +
                "JOIN customer c ON o.custkey = c.custkey " +
                "WHERE o.totalprice > 50000 " +
                "ORDER BY o.totalprice DESC", disabled);
    }

    @Test
    public void testInnerEquiJoinWithFilter()
    {
        Session pushdownEnabled = getSession(true);
        Session disabled = getSession(false);
        assertQueryWithSameQueryRunner(pushdownEnabled, "SELECT o.orderkey, o.totalprice, c.name " +
                "FROM orders o " +
                "JOIN customer c ON o.custkey = c.custkey " +
                "WHERE o.totalprice > 50000 " +
                "ORDER BY o.totalprice DESC", disabled);
    }

    @Test
    public void testSimpleInnerNonEquiJoin()
    {
        Session pushdownEnabled = getSession(true);
        Session disabled = getSession(false);
        assertQueryWithSameQueryRunner(pushdownEnabled, "SELECT o.orderkey, c.name, o.totalprice, c.custkey " +
                "FROM orders o " +
                "JOIN customer c ON o.custkey = c.custkey ", disabled);
    }

    @Test
    public void testInnerNonEquiJoinWithFilter()
    {
        Session pushdownEnabled = getSession(true);
        Session disabled = getSession(false);
        assertQueryWithSameQueryRunner(pushdownEnabled, "SELECT o.orderkey, c.name, o.totalprice, c.custkey " +
                "FROM orders o " +
                "JOIN customer c ON o.custkey = c.custkey " +
                "WHERE o.totalprice < c.acctbal / 2", disabled);
    }

    @Override
    @Test
    public void testShuffledStatsWithInnerJoin()
    {
        // NOTE: Overriding the test case as the query stats will be different with Pushdown enabled
        if (!(getQueryRunner() instanceof DistributedQueryRunner) || getQueryRunner().getDefaultSession().getSystemProperty("spill_enabled", Boolean.class)) {
            return;
        }
        DistributedQueryRunner queryRunner = (DistributedQueryRunner) getQueryRunner();

        String query = "SELECT a.orderkey, a.orderstatus, b.linenumber FROM orders a JOIN lineitem b ON a.orderkey = b.orderkey";
        // Set session property to enforce a hash partitioned join.
        Session partitionedJoin = Session.builder(getSession())
                .setSystemProperty(JOIN_DISTRIBUTION_TYPE, PARTITIONED.name())
                .setSystemProperty(JOIN_REORDERING_STRATEGY, NONE.name())
                .build();
        QueryId partitionQueryId = queryRunner.executeWithQueryId(partitionedJoin, query).getQueryId();
        QueryInfo partitionJoinQueryInfo = queryRunner.getQueryInfo(partitionQueryId);
        assertEquals(partitionJoinQueryInfo.getQueryStats().getOutputPositions(), getTableRowCount("lineitem"));
        // Verify the number shuffled rows, raw input rows and output rows in hash partitioned join.
        // NOTE: the latter two shall be the same for both hash partitioned join and broadcast join.
        assertEquals(partitionJoinQueryInfo.getQueryStats().getRawInputPositions(), getTableRowCount("lineitem"));
        assertEquals(partitionJoinQueryInfo.getQueryStats().getShuffledPositions(), getTableRowCount("lineitem"));

        // Set session property to enforce a broadcast join.
        Session broadcastJoin = Session.builder(getSession())
                .setSystemProperty(JOIN_DISTRIBUTION_TYPE, BROADCAST.name())
                .setSystemProperty(JOIN_REORDERING_STRATEGY, NONE.name())
                .build();

        QueryId broadcastQueryId = queryRunner.executeWithQueryId(broadcastJoin, query).getQueryId();
        assertNotEquals(partitionQueryId, broadcastQueryId);
        QueryInfo broadcastJoinQueryInfo = queryRunner.getQueryInfo(broadcastQueryId);
        assertEquals(broadcastJoinQueryInfo.getQueryStats().getRawInputPositions(), getTableRowCount("lineitem"));
        assertEquals(broadcastJoinQueryInfo.getQueryStats().getOutputPositions(), getTableRowCount("lineitem"));
        // NOTE: the number of shuffled bytes except the final output should be a multiple of the number of rows in lineitem table in broadcast join case.
        assertEquals(((broadcastJoinQueryInfo.getQueryStats().getShuffledPositions() - getTableRowCount("lineitem")) % getTableRowCount("lineitem")), 0);
        assertEquals(broadcastJoinQueryInfo.getQueryStats().getShuffledPositions(), getTableRowCount("lineitem"));
        // Both partitioned join and broadcast join should have the same raw input data set.
        assertEquals(partitionJoinQueryInfo.getQueryStats().getRawInputPositions(), broadcastJoinQueryInfo.getQueryStats().getRawInputPositions());
        assertEquals(partitionJoinQueryInfo.getQueryStats().getShuffledDataSize().toBytes(), broadcastJoinQueryInfo.getQueryStats().getShuffledDataSize().toBytes());
    }

    private Session getSession(boolean pushdownEnabled)
    {
        return Session.builder(getSession())
                .setSystemProperty(INNER_JOIN_PUSHDOWN_ENABLED, String.valueOf(pushdownEnabled))
                .build();
    }
}
