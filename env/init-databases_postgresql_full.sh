#!/bin/bash
set -e # -e means exit the script if any command fails

# Wait for PostgreSQL to be ready
until psql -U root -c '\l' 2>/dev/null; do
  >&2 echo "PostgreSQL is unavailable - waiting..."
  sleep 2
done

echo "PostgreSQL is ready!"

############################
# 0. Create template DB: sql_test_template
############################
echo "Creating template database: sql_test_template with UTF8 encoding (formerly sql_test)"
psql -U root -tc "SELECT 1 FROM pg_database WHERE datname='sql_test_template'" | grep -q 1 \
  || psql -U root -c "CREATE DATABASE sql_test_template WITH OWNER=root ENCODING='UTF8' TEMPLATE=template0;"

# Create schemas in sql_test_template
echo "Creating required schemas: test_schema and test_schema_2 in sql_test_template"
psql -U root -d sql_test_template -c "CREATE SCHEMA IF NOT EXISTS test_schema;"
psql -U root -d sql_test_template -c "CREATE SCHEMA IF NOT EXISTS test_schema_2;"

# Create hstore, citext extensions
echo "Creating hstore and citext extensions in sql_test_template..."
psql -U root -d sql_test_template -c "CREATE EXTENSION IF NOT EXISTS hstore;"
psql -U root -d sql_test_template -c "CREATE EXTENSION IF NOT EXISTS citext;"

# Set default_text_search_config
echo "Setting default_text_search_config to pg_catalog.english in sql_test_template..."
psql -U root -d sql_test_template -c "ALTER DATABASE sql_test_template SET default_text_search_config = 'pg_catalog.english';"

echo "NOTE: For two-phase transaction support, set 'max_prepared_transactions' > 0 in postgresql.conf."

############################
# 1. Define DB → tables mapping
############################
declare -A DATABASE_MAPPING=(
    ["archeology_scan_template"]="projects personnel sites equipment scans environment pointcloud mesh spatial features conservation registration processing qualitycontrol"
    ["sports_events_template"]="circuits constructors drivers races constructor_results constructor_standings driver_standings lap_times pit_stops qualifying sprint_results"
    ["cold_chain_pharma_compliance_template"]="shipments products productbatches carriers vehicles monitoringdevices environmentalmonitoring qualitycompliance incidentandriskmanagement insuranceclaims reviewsandimprovements shipsensorlink"
    ["cross_border_template"]="DataFlow RiskManagement DataProfile SecurityProfile VendorManagement Compliance AuditAndCompliance"
    ["crypto_exchange_template"]="users orders orderExecutions fees marketdata marketstats analyticsindicators riskandmargin accountbalances systemmonitoring Exchange_OrderType_Map"
    ["cybermarket_pattern_template"]="markets vendors buyers products transactions transaction_products vendor_markets vendor_countries vendor_payment_methods communications connection_security risk_analytics alerts"
    ["disaster_relief_template"]="disasterevents distributionhubs operations supplies transportation humanresources financials beneficiariesandassessments environmentandhealth coordinationandevaluation operation_hub_map"    ["exchange_traded_funds_template"]="families exchanges categories sectors bond_ratings securities funds family_categories family_exchanges sector_allocations bond_allocations holdings performance annual_returns risk_metrics"
    ["fake_account_template"]="platforms accounts profiles security_sessions content_activity network_metrics interaction_metrics behavioral_scores risk_and_moderation cluster_analysis account_clusters monitoring"
    ["households_template"]="locations infrastructure service_types households properties transportation_assets amenities"
    ["hulushows_template"]="companies rollups core content_info availabilitys promo_info show_rollups"
    ["insider_trading_template"]="traders instruments trader_relationships order_status_types trade_records market_conditions order_behaviour manipulation_signals sentiment_analytics corporate_events reg_compliance enforcement_actions"
    ["labor_certification_applications_template"]="employer employer_poc attorney preparer worksite prevailing_wage cases case_attorney case_worksite"
    ["mental_health_template"]="facilities clinicians patients assessmentbasics encounters assessmentsymptomsandrisk assessmentsocialanddiagnosis treatmentbasics treatmentoutcomes"
    ["museum_artifact_template"]="ArtifactsCore ArtifactRatings SensitivityData ExhibitionHalls Showcases EnvironmentalReadingsCore AirQualityReadings SurfaceAndPhysicalReadings LightAndRadiationReadings ConditionAssessments RiskAssessments ConservationAndMaintenance UsageRecords ArtifactSecurityAccess Monitor_Showcase_Map"
    ["organ_transplant_template"]="demographics recipients_demographics medical_history hla_info function_and_recovery clinical recipients_immunology transplant_matching compatibility_metrics risk_evaluation allocation_details logistics administrative_and_review data_source_and_quality"
    ["planets_data_template"]="stars instruments_surveys planets orbital_characteristics physical_properties planet_instrument_observations data_quality_tracking"
    ["polar_equipment_template"]="EquipmentType Equipment Location OperationMaintenance PowerBattery EngineAndFluids Transmission ChassisAndVehicle Communication CabinEnvironment LightingAndSafety WaterAndWaste Scientific WeatherAndStructure ThermalSolarWindAndGrid StationEquipmentType"
    ["reverse_logistics_template"]="customers products orders returns quality_assessment return_processing financial_management case_management"
    ["robot_fault_prediction_template"]="robot_record robot_details operation joint_performance joint_condition actuation_data mechanical_status system_controller maintenance_and_fault performance_and_safety"
    ["solar_panel_template"]="panel_models plants plant_panel_model plant_record electrical_performance environmental_conditions mechanical_condition operational_metrics inspection alert"
    ["virtual_idol_template"]="fans virtualidols interactions membershipandspending engagement commerceandcollection socialcommunity eventsandclub loyaltyandachievements preferencesandsettings moderationandcompliance supportandfeedback retentionandinfluence additionalnotes"
)


############################
# 2. Create template DBs and import data
############################
for DB_TEMPLATE in "${!DATABASE_MAPPING[@]}"; do
    echo "Creating template database: $DB_TEMPLATE"
    psql -U root -tc "SELECT 1 FROM pg_database WHERE datname='${DB_TEMPLATE}'" | grep -q 1 \
      || psql -U root -c "CREATE DATABASE ${DB_TEMPLATE} WITH OWNER=root ENCODING='UTF8' TEMPLATE=template0;"
done

# Function to import files from database-specific folders
import_db_files() {
    local db_template="$1"
    local db_folder="/docker-entrypoint-initdb.d/postgre_table_dumps/${db_template}"
    
    echo "Importing files for ${db_template} from ${db_folder}"
    
    # Check if the folder exists
    if [[ ! -d "${db_folder}" ]]; then
        echo "Warning: Folder ${db_folder} does not exist, skipping database ${db_template}"
        return
    fi
    
    # Special case for global_atlas_template
    if [[ "${db_template}" == "global_atlas_template" ]]; then
        # Check if the schema and inputs files exist
        local schema_file="${db_folder}/global_atlas-schema.sql"
        local inputs_file="${db_folder}/global_atlas-inputs.sql"
        
        if [[ -f "${schema_file}" && -f "${inputs_file}" ]]; then
            echo "Importing global_atlas schema file to ${db_template}..."
            psql -U root -d "${db_template}" -f "${schema_file}" 2>>/tmp/error.log \
                || echo "Error importing schema file for ${db_template}. Check /tmp/error.log for details."
            
            echo "Importing global_atlas data file to ${db_template}..."
            psql -U root -d "${db_template}" -f "${inputs_file}" 2>>/tmp/error.log \
                || echo "Error importing data file for ${db_template}. Check /tmp/error.log for details."
        else
            # If the special files don't exist, fall back to importing individual table files
            echo "Special global_atlas files not found, falling back to individual table imports."
            import_table_files "${db_template}" "${db_folder}"
        fi
    else
        # Regular case: import all table files in the folder
        import_table_files "${db_template}" "${db_folder}"
    fi
}

# Function to import individual table files
import_table_files() {
    local db_template="$1"
    local db_folder="$2"
    local tables="${DATABASE_MAPPING[$db_template]}"
    
    for table in $tables; do
        local sql_file="${db_folder}/${table}.sql"
        if [[ -f "$sql_file" ]]; then
            echo "Importing ${sql_file} into database ${db_template}"
            if ! psql -U root -d "${db_template}" -f "${sql_file}" 2>>/tmp/error.log; then
                echo "Error importing ${sql_file} into database ${db_template}. Check /tmp/error.log for details."
            fi
        else
            echo "Error: SQL file ${sql_file} not found for table ${table}"
            # trigger stop
            # exit 1
        fi
    done
}

# Import data for each database
for DB_TEMPLATE in "${!DATABASE_MAPPING[@]}"; do
    import_db_files "${DB_TEMPLATE}"
done

if [[ -s /tmp/error.log ]]; then
    echo "Errors occurred during import:"
    cat /tmp/error.log
fi

rm -f /tmp/error.log

############################
# 3. Mark these template DBs as 'datistemplate = true'
############################
echo "Marking these template databases as 'datistemplate = true'..."
for DB_TEMPLATE in "${!DATABASE_MAPPING[@]}"; do
  psql -U root -d postgres -c "UPDATE pg_database SET datistemplate = true WHERE datname = '${DB_TEMPLATE}';" || true
done

############################
# Example usage
############################
echo "All template databases created. For example, to clone 'financial_template' into 'financial':"
echo "    dropdb financial || true"
echo "    createdb financial --template=financial_template"
echo ""
echo "Done creating template DBs."

echo "Now creating real DB from each template DB..."

for DB_TEMPLATE in "${!DATABASE_MAPPING[@]}"; do
  REAL_DB="${DB_TEMPLATE%_template}"
  echo "Checking if real database '${REAL_DB}' exists..."
  EXISTS=$(psql -U root -tc "SELECT 1 FROM pg_database WHERE datname='${REAL_DB}'" | grep -c 1 || true)
  if [[ "$EXISTS" -eq 0 ]]; then
    echo "Creating real database '${REAL_DB}' from template '${DB_TEMPLATE}'"
    psql -U root -c "CREATE DATABASE ${REAL_DB} WITH OWNER=root TEMPLATE=${DB_TEMPLATE};"
  else
    echo "Database '${REAL_DB}' already exists, skipping creation."
  fi
done

echo "Done creating real DBs."