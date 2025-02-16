
CREATE APPLICATION ROLE IF NOT EXISTS APP_PUBLIC;

CREATE OR ALTER VERSIONED SCHEMA LOGIC;
CREATE OR ALTER SCHEMA OUT;

GRANT USAGE ON SCHEMA OUT TO APPLICATION ROLE APP_PUBLIC;
GRANT USAGE ON SCHEMA LOGIC TO APPLICATION ROLE APP_PUBLIC;


CREATE TABLE if not exists OUT.SETTINGS (brand varchar);
-- prep for later ...
create or replace procedure out.outDB(prefix varchar)
RETURNS VARCHAR
LANGUAGE PYTHON
PACKAGES = ('snowflake-snowpark-python')
RUNTIME_VERSION = 3.9
HANDLER = 'main'
AS $$
def main(session, prefix):
    session.sql(f"""
    create database if not EXISTS DATA_MAPPING_DB_SAMPLE_DB;
        """).collect()
    session.sql(f"""
        grant usage on database DATA_MAPPING_DB_SAMPLE_DB to application role app_public;
        """).collect()

    session.sql(f"""
        grant usage on schema DATA_MAPPING_DB_SAMPLE_DB.PUBLIC to application role app_public;
        """).collect()

    return f"DONE"
$$;

grant usage on procedure out.outDB(varchar) to application role app_public;

create or replace procedure out.sampledb()
    returns string
    language sql
    as $$
        begin
            create database if not EXISTS DATA_MAPPING_DB_SAMPLE_DB;
            grant usage on database DATA_MAPPING_DB_SAMPLE_DB to application role app_public;
            grant usage on schema DATA_MAPPING_DB_SAMPLE_DB.PUBLIC to application role app_public;
            create or replace TABLE  DATA_MAPPING_DB_SAMPLE_DB.PUBLIC.SALES_DATA (
                EMAIL VARCHAR(255),
                DATE TIMESTAMP_NTZ(9),
                SHOP NUMBER(38,0),
                QTY NUMBER(38,0),
                PRICE NUMBER(10,2),
                ITEM_ID VARCHAR(100),
                TRANS_ID VARCHAR(100),
                CURRENCY VARCHAR(10)
            );
            INSERT INTO DATA_MAPPING_DB_SAMPLE_DB.PUBLIC.SALES_DATA 
                (EMAIL, DATE, SHOP, QTY, PRICE, ITEM_ID, TRANS_ID, CURRENCY)  
            VALUES  
                ('customer@example.com', CURRENT_TIMESTAMP, 101, 2, 19.99, 'ITEM123', 'TRANS456', 'USD');  
            grant SELECT,INSERT, REFERENCES, UPDATE on table DATA_MAPPING_DB_SAMPLE_DB.PUBLIC.SALES_DATA to application role app_public;
            return 'SUCCESS';
        end;
    $$;

grant usage on procedure out.sampledb() to application role app_public;

create or replace procedure out.revoke_grants()
    returns string
    language sql
    as $$
        begin
            revoke usage on database DATA_MAPPING_DB_SAMPLE_DB from application role app_public;
            revoke usage on schema DATA_MAPPING_DB_SAMPLE_DB.PUBLIC from application role app_public;
            revoke SELECT,INSERT, REFERENCES, UPDATE on table DATA_MAPPING_DB_SAMPLE_DB.PUBLIC.SALES_DATA from application role app_public;
            return 'SUCCESS';
        end;
    $$;

grant usage on procedure out.revoke_grants() to application role app_public;



--this is the permissions callback we saw in the manifest.yml file
create or replace procedure logic.register_single_callback(ref_name string, operation string, ref_or_alias string)
    returns string
    language sql
    as $$
        begin
            case (operation)
                when 'ADD' then
                    select system$set_reference(:ref_name, :ref_or_alias);
                when 'REMOVE' then
                    select system$remove_reference(:ref_name);
                when 'CLEAR' then
                    select system$remove_reference(:ref_name);
                else
                    return 'Unknown operation: ' || operation;
            end case;
            system$log('debug', 'register_single_callback: ' || operation || ' succeeded');
            return 'Operation ' || operation || ' succeeded';
        end;
    $$;

--grant the application role permissions to the procedure
grant usage on procedure logic.register_single_callback(string, string, string) to application role app_public;

--create a schema for the UI (streamlit)
create or alter versioned schema ui;
--grant the application role permissions onto the schema
grant usage on schema ui to application role app_public;

create streamlit if not exists ui."Data Mapping" from 'ui' main_file='main.py';

--grant the application role permissions onto the streamlit
grant usage on streamlit ui."Data Mapping" TO APPLICATION ROLE APP_PUBLIC;

