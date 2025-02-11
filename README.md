# Data Mapping

## Simple Data Mapping with 3 different technics

- LLM Cortex
- Cosine Similarity
- Manual

## Prerequisites

1. Snowflake CLI
2. Snowflake Account
3. User with DEMO_ROLE (to be created manually)

## Run instructions 


1. Change your brand name in main.py

    ```sh
    CP_NAME='ACME'
    ```

1. Add the logo in /img folder, name the logo in lower case as the brand

    ```sh
    acme.png
    ```

3. Run:

 ```sh
    snow app run --role DEMO_ROLE
    ```

