# EPA Data Notes

The New York State Department of Environmental Conservation (NYSDEC) is responsible for inspecting hazardous waste-producing facilities to ensure compliance with federal regulations. As there are over 50,000 facilities and inspections are very time consuming, it is important to ensure that inspectors spend their time inspecting the facilities with the highest risk of violating these regulations.

For this project, you will have access to public waste shipment and inspection data, available from both the Federal Environmental Protection Agency (EPA) and NYSDEC. While national data has been provided for data from the EPA, this project focuses only on inspections in the state of New York. The provided data sources include:
- **RCRAInfo**: Contains inspection, violation, and enforcement data related to the Resource Conservation and Recovery Act (RCRA), as well as information about facilities and handlers of hazardous waste. These data are available in the `rcra` schema, and note in particular the inspections and results information found in `rcra.cmecomp3` (details about this table can be found in the data dictionary under "Data Element Dictionary" -> "Reporting Tables" -> "CM&E"). For more information, see:
    - [RCRAInfo Data Summary](https://echo.epa.gov/tools/data-downloads/rcrainfo-download-summary)
    - [General Information about RCRA](https://rcrapublic.epa.gov/rcrainfoweb/action/main-menu/view)
    - [Detailed Data Dictionary](https://rcrainfo.epa.gov/rcrainfo-help/application/publicHelp/index.htm#introduction.htm)
- **ICIS-FE&C**: Federal enforcement and compliance (FE&C) data from the Integrated Compliance Information System (ICIS), available in the `fec` schema. [More information here](https://echo.epa.gov/tools/data-downloads/icis-fec-download-summary).
- **FRS**: Data in the `frs` schema is from the Facility Registry Service (FRS), allowing for linking facilities between ICIS and RCRAInfo datasets. [More information here](https://echo.epa.gov/tools/data-downloads/frs-download-summary).
- **ICIS-Air**: Data in the `air` schema is from the Integrated Compliance Information System for Air. [More information here](https://echo.epa.gov/tools/data-downloads/icis-air-download-summary).
- **ICIS-NPDES**: Data in the `npdes` schema is from the Integrated Compliance Information System National Pollutant Discharge Elimination System (NPDES). [More information here](https://echo.epa.gov/tools/data-downloads/icis-npdes-download-summary).
- **NYSDEC Reports**: The `nysdec_reports` schema includes information from reports filed annually by large quantity hazardous waste generators as well as treatment, storage, and disposal facilities in the state of New York. For more information, see:
    - [General information about the reports](https://www.dec.ny.gov/chemical/57604.html)
    - [Reporting forms](https://www.dec.ny.gov/chemical/57619.html)
- **Manifest Data**: The `manifest` schema contains information about hazardous waste shipments to, from, or within the state of New York. More information:
    - [Data files and overview](http://www.dec.ny.gov/chemical/9098.html)
    - [General information about manifests](http://www.dec.ny.gov/chemical/60805.html)
    - [Hazardous waste codes and designations](https://govt.westlaw.com/nycrr/Document/I4eacc3f8cd1711dda432a117e6e0f345?viewType=FullText&originationContext=documenttoc&transitionType=CategoryPageItem&contextData=(sc.Default))

## Schemas

### ``nysdec_reports``
The Hazardous Waste Report is a summary of all hazardous waste generated in the previous calendar year by all large quantity generators (LQG's) as well as all waste received by all treatment, storage or disposal facilities (TSD). 
#### ``gm*``: Form GM (Generation and Management Form)
- describes detailed information about each waste generated at the site
- `gm1`:

    | column | Description |
    | --- | ----------- |
    | ``hz_pg`` | page number (one waste per page) |
    | `form_code`| physical form or chemical composition of the hazardous waste (Section [IX.E.](https://www.dec.ny.gov/docs/materials_minerals_pdf/2019hwinstruct.pdf)) |
    | `unit_of_measure` | {`1`: pounds,`2`: short tons,`3`: kilograms,`4`: metric tones,`5`: gallons,`6`: liters,`7`: cubic yards} |
    | `wst_density`| decimal; val = `1.00` and `density_unit_of_measure` = `2` if density unknown |
    | `density_unit_of_measure`| {`0`: blank, `1`: pounds per gallon (lbs./gal), `2` : specific gravity(sg)} (Note: `0` if `unit_of_measure` is not `5`,`6`, or `7`)| 
    | `source_code`| how the hazardous waste originated (Section [IX.C.](https://www.dec.ny.gov/docs/materials_minerals_pdf/2019hwinstruct.pdf))
    |`managment_method`| **(for Source G25 only)** on-site treatment, disposal, or recycling process system in which the waste was or will be managed (Section [IX.D.](https://www.dec.ny.gov/docs/materials_minerals_pdf/2019hwinstruct.pdf))|
    |`gen_quantity`| quantity generated in reporting year |
    |`waste_min_code`| waste minimization, recycling, or pollution prevention efforts implemented to reduce the volume and toxicity of hazardous waste (Section [IX.F](https://www.dec.ny.gov/docs/materials_minerals_pdf/2019hwinstruct.pdf))|
    |`on_site_mangement`, `off_site_shipment`| either maneged on-site or shipped off-site|
 
- ``gm1nydec`` : Regulatory Fees and Hazardous Waste Regulatory Fee (used to determine if the hazardous waste includes wastewater, which may be subject to regulatory fee and fee exemption)
    
    | column | Description |
    | --- | ----------- |
    | ``wastewater`` | `Y` if this hazardous waste contains: (a) a minimum of 95% water by weight; and (b) a maximum of 1% by weight of total organic carbon; and (c) a maximum of 1%  by weight of total suspended solids (i.e., total filterable solids) |
    | `exempt_residual`, `exempt_recycling`| two types of Regulatory Fee Exemptions|  
 
 - ``gm2`` : EPA waste code for each reported waste generated
 - `gm3` : state waste code for each reported waste generated
 
 - ``gm4``: off-site management information for the reported waste
 
    | column | Description |
    | --- | ----------- |
    | ``hz_pg`` | page number (one waste per page) |
    |`io_pg_num_seq`| off-site sequence number (1-3)|
    |`managment_method`| off-site facility's management method |
    |`io_tdr_id`| EPA ID Number for the facility to which waste was shipped|
    |`io_tdr_qty`| quantity shipped in reporting year|
    
 - ``gm5``: on-site management information for the reported waste
 
    | column | Description |
    | --- | ----------- |
    | ``hz_pg`` | page number (one waste per page) |
    |`sys_pg_num_seq`| on-site process system number (1-2) |
    |`managment_method`| on-site management method |
    |`sys_tdr_qty`| quantity treated, disposed, or recycled on site |

#### ``si*``: Site ID Form (Site Identification Form)
- describes detailed information about this site
- ``si1``: General site Information

    | column | Description |
    | --- | ----------- |
    | ``handler_id`` | handler id |
    | ``state_district`` | [DEC Region Codes](https://www.dec.ny.gov/24.html) |
    | `land_type` | {`P`: Private, `C`: County, `D` : District, `F`: Federal, `T`: Tribal, `M`: Municipal, `S`: State, `O`: Other}       |
    | `short_term_generator`| whether this site generates from a short‐term or one‐time event and not from on‐going processes (all equal to `N`)|
    | `transporter`, `transfer_facility`| whether this site is transport/transfer facility of hazardous waste|
    | `tsd_activity`| Treater, Storer or Disposer of Hazardous Waste | 
    |`importer_activity`|United States Importer of Hazardous Waste|
    | `recycler_activity`| Recycler of Hazardous Waste | 
    |`mixed_waste_generator`| Mixed Waste (hazardous and radioactive) Generator |
    |`onsite_burner_exemption`|  Small Quantity On-site Burner Exemption |
    |`furnace_exemption`| Smelting, Melting, and Furnace Exemption |
    |`underground_injection_activity`| Underground Injection Control |
    |`off_site_receipt`| Receives Hazardous Waste from Off-Site |
    |`used_oil_*`| answers to "Used Oil Activities" questions (Section C) |
    |`subpart_k_college`, `subpart_k_hospital`, `subpart_k_nonprofit`, `subpart_k_withdrawal`| Notification for opting into or withdrawing from managing laboratory (College/University, Teaching Hospital, or Non-profit Institute) hazardous wastes pursuant to 40 CFR 262 Subpart K. |
    
    
- ``si2``: Site Owner/Operator Information

    | column | Description |
    | --- | ----------- |
    | ``owner_operator_seq`` | 1-4, corresponds to the order of entries filled in table (verified that 3-4 are duplicate records) |
    | ``owner_operator_indicator``| `CO` = Owner, `CP` = Operator|
    | ``owner_operator_name``| name of owner/operator company |
    | `date_became_current`| date became owner/operator       |
    | `owner_operator_type`| {`P`: Private, `C`: County, `D` : District, `F`: Federal, `T`: Tribal, `M`: Municipal, `S`: State, `O`: Other}       |
    
- ``si3``: NAICS Codes (1-to-many, each handler may have multiple NAICS codes)
- ``si4``: Waste Codes for _Federally_ Regulated Hazardous Wastes ([EPA Waste Codes](http://www.gecap.org/pdf/hazardouswastecodes.pdf)http://www.gecap.org/pdf/hazardouswastecodes.pdf))
- ``si5``: Waste Codes for _NY State_ Regulated Hazardous Wastes ([NY Waste Codes](https://govt.westlaw.com/nycrr/Document/I4eacc3f8cd1711dda432a117e6e0f345?contextData=%28sc.Default%29&transitionType=Default))
- ``si6``: Response to 'Universal Waste Activities' Questions

    | column | Description |
    | --- | ----------- |
    | ``universal_waste_owner`` | all equal to `HQ` |
    | ``universal waste`` | waste type: {`B` : Batteries, `P`: Pesticides, `M`: Mercury containing equipment, `L`: Lamps } |
    | `generated` | all empty string |
    | `accmulated` | accumulated > 5,000 kg or more of universal waste; all equal to `Y`|
    

- ``si7``: Form's certifier data (certifier's name, title, signature data etc.); 1-to-many: each handler has multiple certifier

#### ``wr*``: Form WR (Waste Received Form)
- describes any hazardous waste received by the site during the reporting year (the waste could be managed onsite or subsequently shipped off-site)
- ``wr1`` :

  | column | Description |
  | --- | ----------- |
  | `unit_of_measure` | {`1`: pounds,`2`: short tons,`3`: kilograms,`4`: metric tones,`5`: gallons,`6`: liters,`7`: cubic yards} |
  | `wst_density`| decimal; val = `1.00` and `density_unit_of_measure` = `2` if density unknown |
  | `density_unit_of_measure`| {`0`: blank, `1`: pounds per gallon (lbs./gal), `2` : specific gravity(sg)} (Note: `0` if `unit_of_measure` is not `5`,`6`, or `7`)| 
  |`form_code`| physical form or chemical composition of the hazardous waste (Section [IX.E.](https://www.dec.ny.gov/docs/materials_minerals_pdf/2019hwinstruct.pdf)) |
  |`io_tdr_id`| off-site source EPA ID Number|
  |`io_tdr_qty`| quantity received in reporting year|
  |`hz_pg`, `sub_pg_num`| page number, waste number (maximum three wastes per page) |
 
 - ``wr2`` : EPA waste code for each reported waste received
 - `wr3` : state waste code for each reported waste received
 
 ___
 ### ``air``
 ICIS-Air contains emissions, compliance, and enforcement data on stationary sources of air pollution. Regulated sources cover a wide spectrum; from large industrial facilities to relatively small operations such as dry cleaners.
 
 #### ``icis_air_facilities``: Facility/Source Level Identifying Data
 
Contains basic information about each facility (including name, address, type of facility, the type of air pollutant, the facility's operating status).
 
 #### ``icis_air_polluants``
 
 Information regarding each type of pollutants
 
 
 #### ``icis_air_fces_pces`` : Air Full Compliance Evaluations (FCEs) and Partial Compliance Evaluations (PCEs) 
 
 Data related to each FCE and PCE.
 
 
 ___
 
#### Manifest

Overall, this is the table keeping track of the movement of hazardous waste. Going through Generator, TSDF (treatment, storage, disposal facility), Transporter, it records the Containers (includes type of containers, number, quantity &amp; units, wastes, handling method)

Some main columns showing in all versions:

- Generator
  - Generator\_rcra\_id
  - Date
- TSDF
  - Tsdf\_rcra\_id
  - Date
- Transporter
  - Transporter\_rcra\_id
  - Date
- Containers
  - Container\_type
  - Number of containers
  - Quantitye (waste ONLY), units
  - Handling type
  - Waste code

1990-2006 (mani90-mani05)

Layout: https://www.dec.ny.gov/data/der/esmart/layout9006.txt

Handling method: [https://www.dec.ny.gov/chemical/23914.html](https://www.dec.ny.gov/chemical/23914.html)

Waste code: [https://www.dec.ny.gov/chemical/60836.html](https://www.dec.ny.gov/chemical/60836.html)

2006-now (mani06-mani17)

Layout &amp; explanation (includes enum of waste code, units of measures etc.): [https://www.dec.ny.gov/chemical/60836.html](https://www.dec.ny.gov/chemical/60836.html)

Others:

- Line\_item\_num: unique id of containers from same manifest ID, same generator, tsdf, transporter
- Handling type code &amp; mgmt\_method\_type\_code: https://www.dec.ny.gov/chemical/23914.html
