#!/usr/bin/env python3
"""
Database Metadata Checker for BIRD-Interact
This script helps verify that PostgreSQL databases are loaded correctly by checking:
- Total number of databases
- Total number of tables per database
- Total number of columns per database
- Database sizes
"""

import psycopg2
import sys
import argparse
from typing import Dict, List, Tuple

# Expected database configurations
EXPECTED_DATABASES_LITE = {
    "archeology": "projects personnel sites equipment scans scanenvironment scanpointcloud scanmesh scanspatial scanfeatures scanconservation scanregistration scanprocessing scanqc",
    "alien": "observatories telescopes signals signalprobabilities signaladvancedphenomena signalclassification signaldecoding signaldynamics researchprocess observationalconditions sourceproperties",
    "cross_db": "dataflow riskmanagement dataprofile securityprofile vendormanagement compliance auditandcompliance",
    "vaccine": "shipments container transportinfo vaccinedetails regulatoryandmaintenance sensordata datalogger",
    "gaming": "testsessions performance deviceidentity mechanical audioandmedia rgb physicaldurability interactionandcontrol",
    "museum": "artifactscore artifactratings sensitivitydata exhibitionhalls showcases environmentalreadingscore airqualityreadings surfaceandphysicalreadings lightandradiationreadings conditionassessments riskassessments conservationandmaintenance usagerecords artifactsecurityaccess",
    "polar": "equipment location operationmaintenance powerbattery engineandfluids transmission chassisandvehicle communication cabinenvironment lightingandsafety waterandwaste scientific weatherandstructure thermalsolarwindandgrid",
    "solar": "plant panel performance electrical environment maintenance inverter alerts",
    "robot": "robot_record robot_details operation joint_performance joint_condition actuation_data mechanical_status system_controller maintenance_and_fault performance_and_safety",
    "virtual": "fans virtualidols interactions membershipandspending engagement commerceandcollection socialcommunity eventsandclub loyaltyandachievements moderationandcompliance preferencesandsettings supportandfeedback retentionandinfluence additionalnotes",
    "mental": "facilities clinicians patients assessmentbasics encounters assessmentsymptomsandrisk assessmentsocialanddiagnosis treatmentbasics treatmentoutcomes",
    "news": "users devices articles recommendations sessions systemperformance interactions interactionmetrics",
    "insider": "trader transactionrecord advancedbehavior sentimentandfundamentals compliancecase investigationdetails enforcementactions",
    "crypto": "users orders orderexecutions fees marketdata marketstats analyticsindicators riskandmargin accountbalances systemmonitoring",
    "fake": "account profile sessionbehavior networkmetrics contentbehavior messaginganalysis technicalinfo securitydetection moderationaction",
    "cybermarket": "markets vendors buyers products transactions communication riskanalysis securitymonitoring investigation",
    "credit": "core_record employment_and_income expenses_and_assets bank_and_transactions credit_and_compliance credit_accounts_and_history",
    "disaster": "disasterevents distributionhubs operations supplies transportation humanresources financials beneficiariesandassessments environmentandhealth coordinationandevaluation"
}

EXPECTED_DATABASES_FULL = {
    "archeology_scan": "projects personnel sites equipment scans environment pointcloud mesh spatial features conservation registration processing qualitycontrol",
    "sports_events": "circuits constructors drivers races constructor_results constructor_standings driver_standings lap_times pit_stops qualifying sprint_results",
    "cold_chain_pharma_compliance": "shipments products productbatches carriers vehicles monitoringdevices environmentalmonitoring qualitycompliance incidentandriskmanagement insuranceclaims reviewsandimprovements shipsensorlink",
    "cross_border": "DataFlow RiskManagement DataProfile SecurityProfile VendorManagement Compliance AuditAndCompliance",
    "crypto_exchange": "users orders orderExecutions fees marketdata marketstats analyticsindicators riskandmargin accountbalances systemmonitoring Exchange_OrderType_Map",
    "cybermarket_pattern": "markets vendors buyers products transactions transaction_products vendor_markets vendor_countries vendor_payment_methods communications connection_security risk_analytics alerts",
    "disaster_relief": "disasterevents distributionhubs operations supplies transportation humanresources financials beneficiariesandassessments environmentandhealth coordinationandevaluation operation_hub_map",
    "exchange_traded_funds": "families exchanges categories sectors bond_ratings securities funds family_categories family_exchanges sector_allocations bond_allocations holdings performance annual_returns risk_metrics",
    "fake_account": "platforms accounts profiles security_sessions content_activity network_metrics interaction_metrics behavioral_scores risk_and_moderation cluster_analysis account_clusters monitoring",
    "households": "locations infrastructure service_types households properties transportation_assets amenities",
    "hulushows": "companies rollups core content_info availabilitys promo_info show_rollups",
    "insider_trading": "traders instruments trader_relationships order_status_types trade_records market_conditions order_behaviour manipulation_signals sentiment_analytics corporate_events reg_compliance enforcement_actions",
    "labor_certification_applications": "employer employer_poc attorney preparer worksite prevailing_wage cases case_attorney case_worksite",
    "mental_health": "facilities clinicians patients assessmentbasics encounters assessmentsymptomsandrisk assessmentsocialanddiagnosis treatmentbasics treatmentoutcomes",
    "museum_artifact": "ArtifactsCore ArtifactRatings SensitivityData ExhibitionHalls Showcases EnvironmentalReadingsCore AirQualityReadings SurfaceAndPhysicalReadings LightAndRadiationReadings ConditionAssessments RiskAssessments ConservationAndMaintenance UsageRecords ArtifactSecurityAccess Monitor_Showcase_Map",
    "organ_transplant": "demographics recipients_demographics medical_history hla_info function_and_recovery clinical recipients_immunology transplant_matching compatibility_metrics risk_evaluation allocation_details logistics administrative_and_review data_source_and_quality",
    "planets_data": "stars instruments_surveys planets orbital_characteristics physical_properties planet_instrument_observations data_quality_tracking",
    "polar_equipment": "EquipmentType Equipment Location OperationMaintenance PowerBattery EngineAndFluids Transmission ChassisAndVehicle Communication CabinEnvironment LightingAndSafety WaterAndWaste Scientific WeatherAndStructure ThermalSolarWindAndGrid StationEquipmentType",
    "reverse_logistics": "customers products orders returns quality_assessment return_processing financial_management case_management",
    "robot_fault_prediction": "robot_record robot_details operation joint_performance joint_condition actuation_data mechanical_status system_controller maintenance_and_fault performance_and_safety",
    "solar_panel": "panel_models plants plant_panel_model plant_record electrical_performance environmental_conditions mechanical_condition operational_metrics inspection alert",
    "virtual_idol": "fans virtualidols interactions membershipandspending engagement commerceandcollection socialcommunity eventsandclub loyaltyandachievements preferencesandsettings moderationandcompliance supportandfeedback retentionandinfluence additionalnotes"
}


def connect_to_database(host: str, port: int = 5432, user: str = "root", password: str = "123123") -> psycopg2.extensions.connection:
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"  # Connect to default database first
        )
        return conn
    except psycopg2.Error as e:
        print(f"❌ Error connecting to database at {host}:{port}: {e}")
        sys.exit(1)


def get_database_list(conn: psycopg2.extensions.connection) -> List[str]:
    """Get list of all databases"""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT datname 
                FROM pg_database 
                WHERE datistemplate = false 
                AND datname NOT IN ('postgres', 'template0', 'template1', 'root', 'sql_test_template')
                ORDER BY datname;
            """)
            databases = [row[0] for row in cursor.fetchall()]
            return databases
    except psycopg2.Error as e:
        print(f"❌ Error getting database list: {e}")
        return []


def check_expected_databases(databases: List[str], expected_mapping: Dict[str, str]) -> Dict:
    """Check which expected databases are present and missing"""
    expected_dbs = set(expected_mapping.keys())
    actual_dbs = set(databases)
    
    present_dbs = expected_dbs.intersection(actual_dbs)
    missing_dbs = expected_dbs - actual_dbs
    extra_dbs = actual_dbs - expected_dbs
    
    return {
        'expected_count': len(expected_dbs),
        'present_count': len(present_dbs),
        'missing_count': len(missing_dbs),
        'extra_count': len(extra_dbs),
        'present_dbs': sorted(present_dbs),
        'missing_dbs': sorted(missing_dbs),
        'extra_dbs': sorted(extra_dbs)
    }


def check_expected_tables(host: str, port: int, user: str, password: str, database: str, expected_tables: str) -> Dict:
    """Check which expected tables are present in a database"""
    expected_table_list = expected_tables.split()
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        with conn.cursor() as cursor:
            # Get actual tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            
            actual_tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        actual_table_set = set(actual_tables)
        expected_table_set = set(expected_table_list)
        
        present_tables = expected_table_set.intersection(actual_table_set)
        missing_tables = expected_table_set - actual_table_set
        extra_tables = actual_table_set - expected_table_set
        
        return {
            'expected_count': len(expected_table_set),
            'present_count': len(present_tables),
            'missing_count': len(missing_tables),
            'extra_count': len(extra_tables),
            'present_tables': sorted(present_tables),
            'missing_tables': sorted(missing_tables),
            'extra_tables': sorted(extra_tables)
        }
        
    except psycopg2.Error as e:
        print(f"❌ Error checking tables for database '{database}': {e}")
        return None


def get_database_metadata(host: str, port: int, user: str, password: str, database: str) -> Dict:
    """Get metadata for a specific database"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        metadata = {
            'database': database,
            'tables': 0,
            'columns': 0,
            'total_rows': 0,
            'avg_rows_per_table': 0,
            'size_bytes': 0,
            'size_mb': 0,
            'table_details': []
        }
        
        with conn.cursor() as cursor:
            # Get table count and column count using information_schema (more compatible)
            cursor.execute("""
                SELECT 
                    table_schema,
                    table_name,
                    CASE 
                        WHEN table_type = 'BASE TABLE' THEN 'table'
                        ELSE table_type
                    END as table_type
                FROM information_schema.tables 
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name;
            """)
            
            tables = cursor.fetchall()
            metadata['tables'] = len(tables)
            
            total_rows = 0
            tables_with_rows = 0
            
            # Get column count for each table
            for schema, table, table_type in tables:
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_schema = %s AND table_name = %s;
                """, (schema, table))
                
                column_count = cursor.fetchone()[0]
                metadata['columns'] += column_count
                
                # Try to get actual row count
                try:
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM "{schema}"."{table}";
                    """)
                    actual_rows = cursor.fetchone()[0]
                except psycopg2.Error:
                    # Fallback to estimated rows if direct count fails
                    try:
                        cursor.execute("""
                            SELECT COALESCE(n_tup_ins, 0) as estimated_rows
                            FROM pg_stat_user_tables 
                            WHERE schemaname = %s AND relname = %s;
                        """, (schema, table))
                        row_result = cursor.fetchone()
                        actual_rows = row_result[0] if row_result else 0
                    except psycopg2.Error:
                        actual_rows = 0
                
                if actual_rows > 0:
                    total_rows += actual_rows
                    tables_with_rows += 1
                
                metadata['table_details'].append({
                    'schema': schema,
                    'table': table,
                    'columns': column_count,
                    'estimated_rows': actual_rows,
                    'table_type': table_type
                })
            
            metadata['total_rows'] = total_rows
            metadata['avg_rows_per_table'] = round(total_rows / tables_with_rows, 2) if tables_with_rows > 0 else 0
            
            # Get database size
            cursor.execute("SELECT pg_size_pretty(pg_database_size(%s))", (database,))
            size_pretty = cursor.fetchone()[0]
            
            cursor.execute("SELECT pg_database_size(%s)", (database,))
            size_bytes = cursor.fetchone()[0]
            
            metadata['size_bytes'] = size_bytes
            metadata['size_mb'] = round(size_bytes / (1024 * 1024), 2)
            metadata['size_pretty'] = size_pretty
        
        conn.close()
        return metadata
        
    except psycopg2.Error as e:
        print(f"❌ Error getting metadata for database '{database}': {e}")
        return None


def print_metadata_summary(host: str, port: int, metadata_list: List[Dict], expected_mapping: Dict[str, str] = None):
    """Print a summary of database metadata"""
    print(f"\n📊 Database Metadata Summary for {host}:{port}")
    print("=" * 60)
    
    total_databases = len(metadata_list)
    total_tables = sum(m['tables'] for m in metadata_list if m)
    total_columns = sum(m['columns'] for m in metadata_list if m)
    total_rows = sum(m['total_rows'] for m in metadata_list if m)
    total_size_mb = sum(m['size_mb'] for m in metadata_list if m)
    
    # Calculate overall average rows per table (only counting tables with data)
    total_tables_with_data = 0
    for m in metadata_list:
        if m and m['table_details']:
            tables_with_rows = sum(1 for t in m['table_details'] if t['estimated_rows'] > 0)
            total_tables_with_data += tables_with_rows
    
    overall_avg_rows = round(total_rows / total_tables_with_data, 2) if total_tables_with_data > 0 else 0
    
    print(f"📈 Total Databases: {total_databases}")
    print(f"📋 Total Tables: {total_tables}")
    print(f"📋 Tables with Data: {total_tables_with_data}")
    print(f"🔢 Total Columns: {total_columns}")
    print(f"📊 Total Rows: {total_rows:,}")
    print(f"📈 Avg Rows per Table: {overall_avg_rows:,}")
    print(f"💾 Total Size: {total_size_mb:.2f} MB")
    
    # Check against expected databases if mapping provided
    if expected_mapping:
        databases = [m['database'] for m in metadata_list if m]
        db_check = check_expected_databases(databases, expected_mapping)
        
        print(f"\n🎯 Expected Database Check:")
        print(f"   Expected: {db_check['expected_count']}")
        print(f"   Present: {db_check['present_count']} ✅")
        print(f"   Missing: {db_check['missing_count']} ❌")
        print(f"   Extra: {db_check['extra_count']} ⚠️")
        
        if db_check['missing_dbs']:
            print(f"   Missing databases: {', '.join(db_check['missing_dbs'])}")
        if db_check['extra_dbs']:
            print(f"   Extra databases: {', '.join(db_check['extra_dbs'])}")
    
    print()
    
    # # Print details for each database
    # for metadata in metadata_list:
    #     if metadata:
    #         print(f"🗄️  Database: {metadata['database']}")
    #         print(f"   Tables: {metadata['tables']}")
    #         print(f"   Columns: {metadata['columns']}")
    #         print(f"   Rows: {metadata['total_rows']:,}")
    #         print(f"   Avg Rows/Table: {metadata['avg_rows_per_table']:,}")
    #         print(f"   Size: {metadata['size_pretty']}")
            
    #         # Check expected tables if mapping provided
    #         if expected_mapping and metadata['database'] in expected_mapping:
    #             table_check = check_expected_tables(host, port, "root", "123123", 
    #                                               metadata['database'], 
    #                                               expected_mapping[metadata['database']])
    #             if table_check:
    #                 print(f"   Expected Tables: {table_check['expected_count']}")
    #                 print(f"   Present Tables: {table_check['present_count']} ✅")
    #                 print(f"   Missing Tables: {table_check['missing_count']} ❌")
    #                 if table_check['missing_tables']:
    #                     print(f"   Missing: {', '.join(table_check['missing_tables'])}")
            
    #         # Show top 5 tables by row count
    #         if metadata['table_details']:
    #             sorted_tables = sorted(metadata['table_details'], 
    #                                  key=lambda x: x['estimated_rows'], reverse=True)
    #             print(f"   Top tables by rows:")
    #             for table_info in sorted_tables[:5]:
    #                 if table_info['estimated_rows'] > 0:
    #                     print(f"     - {table_info['schema']}.{table_info['table']}: "
    #                           f"{table_info['estimated_rows']:,} rows, {table_info['columns']} columns")
    #         print()


def print_detailed_table_info(metadata_list: List[Dict], show_all: bool = False):
    """Print detailed table information"""
    print("\n📋 Detailed Table Information")
    print("=" * 60)
    
    for metadata in metadata_list:
        if metadata and metadata['table_details']:
            print(f"\n🗄️  Database: {metadata['database']}")
            print("-" * 40)
            
            tables_to_show = metadata['table_details'] if show_all else metadata['table_details'][:10]
            
            for table_info in tables_to_show:
                print(f"  📊 {table_info['schema']}.{table_info['table']}")
                print(f"     Columns: {table_info['columns']}")
                if table_info['estimated_rows'] > 0:
                    print(f"     Estimated Rows: {table_info['estimated_rows']}")
                print(f"     Type: {table_info['table_type']}")
            
            if not show_all and len(metadata['table_details']) > 10:
                print(f"     ... and {len(metadata['table_details']) - 10} more tables")


def main():
    parser = argparse.ArgumentParser(description="Check BIRD-Interact database metadata")
    parser.add_argument("--host", default="bird_interact_postgresql", 
                       help="Database host (default: bird_interact_postgresql)")
    parser.add_argument("--port", type=int, default=5432, 
                       help="Database port (default: 5432)")
    parser.add_argument("--user", default="root", 
                       help="Database user (default: root)")
    parser.add_argument("--password", default="123123", 
                       help="Database password (default: 123123)")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed table information")
    parser.add_argument("--all-tables", action="store_true", 
                       help="Show all tables (use with --detailed)")
    parser.add_argument("--version", choices=["lite", "full"], 
                       help="Specify version to check against expected databases (lite/full)")
    
    args = parser.parse_args()
    
    print(f"🔍 Checking database metadata for {args.host}:{args.port}")
    
    # Determine expected mapping based on host or version
    expected_mapping = None
    if args.version:
        expected_mapping = EXPECTED_DATABASES_LITE if args.version == "lite" else EXPECTED_DATABASES_FULL
    elif "postgresql_full" in args.host:
        expected_mapping = EXPECTED_DATABASES_FULL
    elif "postgresql" in args.host:
        expected_mapping = EXPECTED_DATABASES_LITE
    
    # Connect to database
    conn = connect_to_database(args.host, args.port, args.user, args.password)
    
    # Get list of databases
    databases = get_database_list(conn)
    conn.close()
    
    if not databases:
        print("❌ No databases found!")
        return
    
    print(f"✅ Found {len(databases)} databases: {', '.join(databases)}")
    
    # Get metadata for each database
    metadata_list = []
    for db in databases:
        print(f"📊 Analyzing database: {db}")
        metadata = get_database_metadata(args.host, args.port, args.user, args.password, db)
        metadata_list.append(metadata)
    
    # Print summary
    print_metadata_summary(args.host, args.port, metadata_list, expected_mapping)
    
    # Print detailed info if requested
    if args.detailed:
        print_detailed_table_info(metadata_list, args.all_tables)
    
    print("\n✅ Database metadata check completed!")
    
    # Provide some guidance
    if expected_mapping:
        expected_total_tables = sum(len(tables.split()) for tables in expected_mapping.values())
        print(f"\n💡 Expected Results for {args.version or 'detected version'}:")
        print(f"   - Expected databases: {len(expected_mapping)}")
        print(f"   - Expected total tables: {expected_total_tables}")
        print("   - If you see significantly fewer tables or rows, check Docker logs for errors")
        print("   - Average rows per table should be > 0 for populated databases")


if __name__ == "__main__":
    main()
