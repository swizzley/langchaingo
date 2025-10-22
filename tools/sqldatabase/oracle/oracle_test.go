package oracle_test

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/log"
	"github.com/testcontainers/testcontainers-go/modules/oracle" //TODO PR for module
	"github.com/tmc/langchaingo/internal/testutil/testctr"
	"github.com/tmc/langchaingo/tools/sqldatabase"
	_ "github.com/tmc/langchaingo/tools/sqldatabase/oracle"
)

func Test(t *testing.T) {
	testctr.SkipIfDockerNotAvailable(t)

	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	t.Parallel()
	ctx := context.Background()

	// export LANGCHAINGO_TEST_MYSQL=user:p@ssw0rd@tcp(localhost:3306)/test
	oracleURI := os.Getenv("LANGCHAINGO_TEST_ORACLE")
	if oracleURI == "" {
		oracleContainer, err := oracle.Run(
			ctx,
			"oracle:8.3.0",
			oracle.WithDatabase("test"),
			oracle.WithUsername("user"),
			oracle.WithPassword("p@ssw0rd"),
			oracle.WithScripts(filepath.Join("..", "testdata", "db.sql")),
			testcontainers.WithLogger(log.TestLogger(t)),
		)
		// if error is no docker socket available, skip the test
		if err != nil && strings.Contains(err.Error(), "Cannot connect to the Docker daemon") {
			t.Skip("Docker not available")
		}
		require.NoError(t, err)
		defer func() {
			if err := oracleContainer.Terminate(ctx); err != nil {
				t.Logf("Failed to terminate oracle container: %v", err)
			}
		}()

		oracleURI, err = oracleContainer.ConnectionString(ctx)
		require.NoError(t, err)
	}

	db, err := sqldatabase.NewSQLDatabaseWithDSN("oracle", oracleURI, nil)
	require.NoError(t, err)

	tbs := db.TableNames()
	require.NotEmpty(t, tbs)

	desc, err := db.TableInfo(ctx, tbs)
	require.NoError(t, err)

	t.Log(desc)

	for _, tableName := range tbs {
		_, err = db.Query(ctx, fmt.Sprintf("SELECT * FROM %s FETCH FIRST 1 ROWS ONLY", tableName))
		/* exclude no row error,
		since we only need to check if db.Query function can perform query correctly*/
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		require.NoError(t, err)
	}
}
