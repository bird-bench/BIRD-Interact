# db_utils.py
import os
import subprocess
import psycopg2
from psycopg2 import OperationalError
from psycopg2.pool import SimpleConnectionPool
from src.config.db_config import get_db_config
from src.envs.bird_interact_env.test_case_utils.logger import (
    log_section_header,
    log_section_footer,
    PrintLogger,
    NullLogger
)
import sys


_postgresql_pools = {}

# Use centralized database configuration

def _get_or_init_pool(db_name):
    """
    Returns a connection pool for the given database name, creating one if it does not exist.
    """
    if db_name not in _postgresql_pools:
        config = get_db_config().copy()
        config.update({"dbname": db_name})
        _postgresql_pools[db_name] = SimpleConnectionPool(
            config["minconn"],
            config["maxconn"],
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
            host=config["host"],
            port=config["port"],
        )
    return _postgresql_pools[db_name]

def perform_query_on_postgresql_databases(query, db_name, conn=None):
    """
    Executes the given query on the specified database, returns (result, conn).
    Automatically commits if the query is recognized as a write operation.
    """
    MAX_ROWS = 10000
    pool = _get_or_init_pool(db_name)
    need_to_put_back = False

    if conn is None:
        conn = pool.getconn()
        need_to_put_back = True

    cursor = conn.cursor()
    cursor.execute("SET statement_timeout = '60s';")  # 60s query timeout

    try:
        cursor.execute(query)
        lower_q = query.strip().lower()
        conn.commit()

        if lower_q.startswith("select") or lower_q.startswith("with"):
            # Fetch up to MAX_ROWS + 1 to see if there's an overflow
            rows = cursor.fetchmany(MAX_ROWS + 1)
            if len(rows) > MAX_ROWS:
                rows = rows[:MAX_ROWS]
            result = rows
        else:
            try:
                result = cursor.fetchall()
            except psycopg2.ProgrammingError:
                result = None

        return (result, conn)

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        if need_to_put_back:
            # If you only need a single query, you could return it right away:
            # But usually, we keep the same conn for subsequent queries, so do nothing.
            # If you truly do not want to reuse it, uncomment below:
            # pool.putconn(conn)
            pass

def close_postgresql_connection(db_name, conn):
    """
    Release a connection back to the pool when you are done with it.
    """
    if db_name in _postgresql_pools:
        pool = _postgresql_pools[db_name]
        pool.putconn(conn)

def close_all_postgresql_pools():
    """
    Closes all connections in all pools (e.g., at application shutdown).
    """
    for pool in _postgresql_pools.values():
        pool.closeall()
    _postgresql_pools.clear()

def close_postgresql_pool(db_name):
    """
    Close the pool for a specific db_name and remove its reference.
    """
    if db_name in _postgresql_pools:
        pool = _postgresql_pools.pop(db_name)
        pool.closeall()

def get_connection_for_phase(db_name, logger):
    """
    Acquire a new connection (borrowed from the connection pool) for a specific phase.
    """
    logger.info(f"Acquiring dedicated connection for phase on db: {db_name}")
    result, conn = perform_query_on_postgresql_databases("SELECT 1", db_name, conn=None)
    return conn

def reset_and_restore_database(db_name, pg_password, logger=None):
    """
    Resets the specified database by creating it from the corresponding template database.
    Steps:
    1) Close the connection pool
    2) Terminate all connections
    3) dropdb
    4) createdb --template
    """
    if logger is None:
        logger = NullLogger()
    try:
        pg_host = get_db_config()['host']
        pg_port = get_db_config()['port']
        pg_user = get_db_config()['user']

        env_vars = os.environ.copy()
        env_vars["PGPASSWORD"] = pg_password

        template_db_name = f"{db_name}_template"

        logger.info(f"Resetting database {db_name} using template {template_db_name}")
        logger.info(f"Closing connection pool for database {db_name} before resetting.")
        close_postgresql_pool(db_name)

        terminate_command = [
            "psql",
            "-h", pg_host,
            "-p", str(pg_port),
            "-U", pg_user,
            "-d", "postgres",
            "-c",
            f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
            """
        ]
        subprocess.run(
            terminate_command,
            check=True,
            env=env_vars,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"All connections to database {db_name} have been terminated.")

        drop_command = [
            "dropdb",
            "--if-exists",
            "-h", pg_host,
            "-p", str(pg_port),
            "-U", pg_user,
            db_name,
        ]
        subprocess.run(
            drop_command,
            check=True,
            env=env_vars,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Database {db_name} dropped if it existed.")

        create_command = [
            "createdb",
            "-h", pg_host,
            "-p", str(pg_port),
            "-U", pg_user,
            db_name,
            "--template", template_db_name
        ]
        subprocess.run(
            create_command,
            check=True,
            env=env_vars,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Database {db_name} created from template {template_db_name} successfully.")

    except subprocess.TimeoutExpired as e:
        logger.error(f"Timeout expired while resetting {db_name} from template: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error resetting {db_name} from template: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error resetting {db_name} from template: {e}")
        sys.exit(1)

def create_ephemeral_db_copies(base_db_names, num_copies, pg_password, logger):
    """
    For each base database in base_db_names, create `num_copies` ephemeral DB copies 
    from base_db_template. Return a dict: {base_db: [ephemeral1, ephemeral2, ...], ...}
    """
    pg_host = get_db_config()['host']
    pg_port = get_db_config()['port']
    pg_user = get_db_config()['user']
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = pg_password

    ephemeral_db_pool = {}

    for base_db in base_db_names:
        base_template = f"{base_db}_template"
        ephemeral_db_pool[base_db] = []

        for i in range(1, num_copies+1):
            ephemeral_name = f"{base_db}_process_{i}"
            # If it already exists, drop it first
            drop_cmd = [
                "dropdb", "--if-exists",
                "-h", pg_host,
                "-p", str(pg_port),
                "-U", pg_user,
                ephemeral_name
            ]
            subprocess.run(drop_cmd, check=False, env=env_vars,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # createdb
            create_cmd = [
                "createdb",
                "-h", pg_host,
                "-p", str(pg_port),
                "-U", pg_user,
                ephemeral_name,
                "--template", base_template
            ]
            logger.info(f"Creating ephemeral db {ephemeral_name} from {base_template}...")
            subprocess.run(create_cmd, check=True, env=env_vars,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            ephemeral_db_pool[base_db].append(ephemeral_name)

        logger.info(f"For base_db={base_db}, ephemeral db list = {ephemeral_db_pool[base_db]}")

    return ephemeral_db_pool

def drop_ephemeral_dbs(ephemeral_db_pool_dict, pg_password, logger):
    """
    Delete all ephemeral databases created during the script execution.
    """
    pg_host = get_db_config()['host']
    pg_port = get_db_config()['port']
    pg_user = get_db_config()['user']
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = pg_password

    logger.info("=== Cleaning up ephemeral databases ===")
    for base_db, ephemeral_list in ephemeral_db_pool_dict.items():
        for ephemeral_db in ephemeral_list:
            logger.info(f"Dropping ephemeral db: {ephemeral_db}")
            drop_cmd = [
                "dropdb",
                "--if-exists",
                "-h", pg_host,
                "-p", str(pg_port),
                "-U", pg_user,
                ephemeral_db
            ]
            try:
                subprocess.run(drop_cmd, check=True, env=env_vars,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to drop ephemeral db {ephemeral_db}: {e}")


def execute_queries(queries, db_name, conn=None, logger=None, section_title="", is_solution=True):
    """
    Executes a list of queries using the same connection (conn).
    Inputs:
        queries: List[str] or str
        db_name: str
        conn: psycopg2.extensions.connection or None
        logger: Logger or None
        section_title: str
        is_solution: bool
    Returns:
        Tuple[List[Any], bool, bool]:
            - query_result: List of query results
            - execution_error: True if there was an error executing the query
            - timeout_error: True if the query timed out
    """
    if logger is None:
        # logger = PrintLogger()
        logger = NullLogger()

    log_section_header(section_title, logger)
    query_result = None
    execution_error = False
    timeout_error = False
    
    if isinstance(queries, str):
        queries = [queries]

    for i, query in enumerate(queries):
        try:
            logger.info(f"Executing query {i+1}/{len(queries)}: {query}")
            query_result, conn = perform_query_on_postgresql_databases(query, db_name, conn=conn)
            logger.info(f"Query result: {query_result}")
        except psycopg2.errors.QueryCanceled as e:
            logger.error(f"Timeout error executing query {i}: {e}")
            if is_solution:
                timeout_error = True
        except OperationalError as e:
            logger.error(f"OperationalError executing query {i}: {e}")
            if is_solution:
                execution_error = str(e)
        except psycopg2.Error as e:
            logger.error(f"psycopg2 Error executing query {i}: {e}")
            if is_solution:
                execution_error = str(e)
        except subprocess.TimeoutExpired as e:
            logger.error(f"Subprocess timeout executing query {i}: {e}")
            if is_solution:
                timeout_error = True
        except Exception as e:
            logger.error(f"Generic error executing query {i}: {e}")
            if is_solution:
                execution_error = str(e)
        finally:
            logger.info(f"[{section_title}] DB: {db_name}, conn info: {conn}")

    log_section_footer(logger)
    return query_result, execution_error, timeout_error