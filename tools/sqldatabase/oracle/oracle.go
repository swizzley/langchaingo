package oracle

import (
	"context"
	"database/sql"
	"fmt"
	_ "github.com/sijms/go-ora/v2" // oracle driver
	"github.com/swizzley/langchaingo/tools/sqldatabase"
	"strings"
)

const EngineName = "oracle"

//nolint:gochecknoinits
func init() {
	sqldatabase.RegisterEngine(EngineName, NewOracle)
}

var _ sqldatabase.Engine = Oracle{}

// Oracle is a Oracle engine.
type Oracle struct {
	db     *sql.DB
	schema string
}

// NewOracle creates a new Oracle engine.
// The dsn is the data source name.(e.g.  oracle://user:pass@host:port/serviceName
// The schema is for being able to switch user schemas from a single user
func NewOracle(dsn string, schema string) (sqldatabase.Engine, error) { //nolint:ireturn
	db, err := sql.Open(EngineName, dsn)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1) //nolint:gomnd

	return &Oracle{
		db:     db,
		schema: schema,
	}, nil
}

func (m Oracle) Dialect() string {
	return EngineName
}

func (m Oracle) Query(ctx context.Context, query string, args ...any) ([]string, [][]string, error) {
	rows, err := m.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, nil, err
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	cols, err := rows.Columns()
	if err != nil {
		return nil, nil, err
	}
	results := make([][]string, 0)
	for rows.Next() {
		row := make([]string, len(cols))
		rowNullable := make([]sql.NullString, len(cols))
		rowPtrs := make([]interface{}, len(cols))
		for i := range row {
			rowPtrs[i] = &rowNullable[i]
		}
		err = rows.Scan(rowPtrs...)
		if err != nil {
			return nil, nil, err
		}
		for i := range rowNullable {
			row[i] = rowNullable[i].String
		}
		results = append(results, row)
	}
	return cols, results, nil
}

func (m Oracle) TableNames(ctx context.Context) ([]string, error) {
	//include type as value prefix
	_, result, err := m.Query(ctx, fmt.Sprintf("select OBJECT_TYPE || ':' || OBJECT_NAME as TYPE_AND_NAME from ALL_OBJECTS where OBJECT_TYPE in ('TABLE', 'VIEW') and OWNER = '%s'", m.schema))
	if err != nil {
		return nil, err
	}

	ret := make([]string, 0, len(result))
	for _, row := range result {
		ret = append(ret, row[0])
	}
	return ret, nil
}

func (m Oracle) TableInfo(ctx context.Context, table string) (string, error) {
	//accommodate views by splitting the value and passing the type as an argument
	tableType := "TABLE"
	tableName := table
	if strings.Contains(tableName, ":") {
		tableType = strings.Split(table, ":")[0]
		tableName = strings.Split(table, ":")[1]
	}
	_, result, err := m.Query(ctx, fmt.Sprintf("SELECT DBMS_METADATA.GET_DDL('%s', '%s', '%s') FROM DUAL", tableType, tableName, m.schema))
	if err != nil {
		return "", err
	}
	if len(result) == 0 {
		return "", sqldatabase.ErrTableNotFound
	}
	if len(result[0]) < 1 { //nolint:gomnd
		return "", sqldatabase.ErrInvalidResult
	}

	return result[0][0], nil //nolint:gomnd
}

func (m Oracle) Close() error {
	return m.db.Close()
}
