import requests
import xmltodict
import pandas as pd
import pprint
#https://gis.stackexchange.com/questions/422396/batch-download-data-from-ssurgo
def soil(lat, lon):
    lonLat = "{0} {1}".format(lon, lat)
    url="https://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx"
    #headers = {'content-type': 'application/soap+xml'}
    headers = {'content-type': 'text/xml'}
    body = """<?xml version="1.0" encoding="utf-8"?>
              <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:sdm="http://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx">
       <soap:Header/>
       <soap:Body>
          <sdm:RunQuery>
             <sdm:Query>SELECT co.cokey as cokey, wtenthbar_r,
                        wthirdbar_r,
                        (soimoistdept_r-soimoistdepb_r) as wetting_front,
                        (dbthirdbar_r-wthirdbar_r)/100, ch.chkey as chkey, comppct_r as prcent, slope_r, hydgrp, runoff, erocl,pi_r,slopelenusle_r,  slope_h as slope, hzname, hzdept_r as deptht, hzdepb_r as depthb, awc_r as awc,
                        claytotal_r as clay, silttotal_r as silt,  partdensity, sandtotal_r as sand, om_r as OM, dbthirdbar_r as bulk_density, wthirdbar_r as th33, ph1to1h2o_r as pH, ksat_r as sat_hidric_cond,
                        (dbthirdbar_r-wthirdbar_r)/100 as bd FROM sacatalog sc
                        FULL OUTER JOIN legend lg  ON sc.areasymbol=lg.areasymbol
                        FULL OUTER JOIN mapunit mu ON lg.lkey=mu.lkey
                        FULL OUTER JOIN component co ON mu.mukey=co.mukey
                        FULL OUTER JOIN comonth cm  ON co.cokey=cm.cokey
                        FULL OUTER JOIN cosoilmoist csm  ON cm.comonthkey=csm.comonthkey
                        FULL OUTER JOIN chorizon ch ON co.cokey=ch.cokey
                        FULL OUTER JOIN chtexturegrp ctg ON ch.chkey=ctg.chkey
                        FULL OUTER JOIN chtexture ct ON ctg.chtgkey=ct.chtgkey
                        FULL OUTER JOIN copmgrp pmg ON co.cokey=pmg.cokey
                        FULL OUTER JOIN corestrictions rt ON co.cokey=rt.cokey
                        WHERE mu.mukey IN (SELECT * from SDA_Get_Mukey_from_intersection_with_WktWgs84('point(""" + lonLat + """)')) order by co.cokey, ch.chkey, prcent, deptht
            </sdm:Query>
          </sdm:RunQuery>
       </soap:Body>
    </soap:Envelope>"""

    response = requests.post(url,data=body,headers=headers)
    # Put query results in dictionary format
    my_dict = xmltodict.parse(response.content)
    # Convert from dictionary to dataframe format

    soil_df = pd.DataFrame.from_dict(my_dict['soap:Envelope']['soap:Body']['RunQueryResponse']['RunQueryResult']['diffgr:diffgram']['NewDataSet']['Table'])
    return soil_df

frames = []
coords = [[37.54189, -120.96683], [37.9014, -121.1895]]
for coord in coords: #For each coordinate pair, create a df and append to frames list
    frames.append(soil(*coord))

df_soil=soil(*[47.2707,-122.5337]).iloc[0]
# Create a new DataFrame with the combined series as the first row
df = pd.DataFrame([df_soil[3:31].values], columns=df_soil[3:31].index)
bulk_density = df['bulk_density']
slope = df['slope']
# Standardize and convert df['hzname'] to numerical horizon values

import re
import numpy as np 
def hzname_to_numeric(hzname):
    """
    Standardize and convert a soil horizon name string to a numeric value.
    """
    if pd.isnull(hzname):
        return np.nan
    # Remove all non-uppercase letters
    hz = re.sub(r'[^A-Z]', '', str(hzname).upper())
    # Map special horizon codes first
    special_map = {
        'BE': 3.5, 'BC': 4.5, 'AC': 3.5, 'EB': 3.5, 'AB': 3.0, 'AE': 2.5
    }
    if hz in special_map:
        return special_map[hz]
    # Map single horizon letters
    single_map = {
        'O': 1.0, 'H': 1.0, 'A': 2.0, 'E': 3.0, 'B': 4.0, 'C': 5.0
    }
    # If matches a single letter, return its value
    if hz in single_map:
        return single_map[hz]
    # If it's a combination, try to match the first valid horizon letter
    for k, v in single_map.items():
        if hz.startswith(k):
            return v
    # If nothing matches, return NaN
    return np.nan

# Apply the conversion to df['hzname'] and create a new column with numeric values
df['hzname_numeric'] = df['hzname'].apply(hzname_to_numeric)