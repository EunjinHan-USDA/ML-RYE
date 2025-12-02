#Note: 
import numpy as np
import requests
import json
import xmltodict
import pandas as pd
import math
import os

lon = -76.900000  #'FSP
lat = 39.03000
point = 'test_site'

lonLat = str(lon) + " " + str(lat)
url="https://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx"
headers = {'content-type': 'text/xml'}
body = """<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:sdm="http://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx">
        <soap:Header/>
        <soap:Body>
            <sdm:RunQuery>
                <sdm:Query>SELECT co.cokey as cokey, ch.chkey as chkey, comppct_r as prcent, slope_r, slope_h as slope, hzname, hzdepb_r as depth, 
                            awc_r as awc, claytotal_r as clay, silttotal_r as silt, sandtotal_r as sand, om_r as OM, dbthirdbar_r as dbthirdbar, 
                            wthirdbar_r/100 as TH33, (dbthirdbar_r-(wthirdbar_r/100)) as bd, wfifteenbar_r/100 as TH1500, wsatiated_r/100 as thm, 
                            ksat_r as Ks FROM sacatalog sc
                            FULL OUTER JOIN legend lg  ON sc.areasymbol=lg.areasymbol
                            FULL OUTER JOIN mapunit mu ON lg.lkey=mu.lkey
                            FULL OUTER JOIN component co ON mu.mukey=co.mukey
                            FULL OUTER JOIN chorizon ch ON co.cokey=ch.cokey
                            FULL OUTER JOIN chtexturegrp ctg ON ch.chkey=ctg.chkey
                            FULL OUTER JOIN chtexture ct ON ctg.chtgkey=ct.chtgkey
                            FULL OUTER JOIN copmgrp pmg ON co.cokey=pmg.cokey
                            FULL OUTER JOIN corestrictions rt ON co.cokey=rt.cokey
                            WHERE mu.mukey IN (SELECT * from SDA_Get_Mukey_from_intersection_with_WktWgs84('point(""" + lonLat + """)')) order by co.cokey, ch.chkey, prcent, depth
                </sdm:Query>
            </sdm:RunQuery>
            </soap:Body>
            </soap:Envelope>"""
## Definitions are from SSURGO2.3.2 Table Column Descriptions
#dbthirdbar_r: The oven dry weight of the less than 2 mm soil material per unit volume of soil at a water tension of 1/3 bar.
#wthirdbar_r: The volumetric content of soil water retained at a tension of 1/3 bar (33 kPa), expressed as a percentage of the whole soil. => TH33
#wfifteenbar_r: The volumetric content of soil water retained at a tension of 15 bars (1500 kPa), expressed as a percentage of the whole soil. => TH1500
#wsatiated_r: The estimated volumetric soil water content at or near zero bar tension, expressed as a percentage of the whole soil.  => thm in ExcelInterface (cf. ths is less than thm)
#ksat_r: The amount of water that would move vertically through a unit area of saturated soil in unit time under unit hydraulic gradient.  => Ks => cm/day

response = requests.post(url,data=body,headers=headers)
        # Put query results in dictionary format
my_dict = xmltodict.parse(response.content)
            
#try:
soil_df = pd.DataFrame.from_dict(my_dict['soap:Envelope']['soap:Body']['RunQueryResponse']['RunQueryResult']['diffgr:diffgram']['NewDataSet']['Table'])

# Drop columns where all values are None or NaN
soil_df = soil_df.dropna(axis=1, how='all')
soil_df = soil_df[soil_df.chkey.notnull()]

# Drop unecessary columns
soil_df = soil_df.drop(['@diffgr:id', '@msdata:rowOrder', '@diffgr:hasChanges'], axis=1)

# Drop duplicate rows
soil_df = soil_df.drop_duplicates()

# Convert prcent and depth column from object to float to create 2DSOIL input file
soil_df['prcent'] = soil_df['prcent'].astype(float)
soil_df['soilFile'] = point + '.soi'
soil_df['Bottom depth'] = soil_df['depth'].astype(float)
soil_df['Init Type'] = '\'m\''
soil_df['NO3 (ppm)'] = 10   
soil_df['NH4'] = 0.0
soil_df['HNew'] = -200
soil_df['Tmpr'] = 25.0 
# soil_df['TH1500'] = -1
soil_df['thr'] = -1
soil_df['ths'] = -1
soil_df['tha'] = -1
# soil_df['thm'] = -1
soil_df['Alfa'] = -1
soil_df['n'] = -1
# soil_df['Ks'] = -1
soil_df['Kk'] = -1
soil_df['thk'] = -1

soil_df['Humus_C'] = -1
soil_df['Humus_N'] = -1
soil_df['Litter_C'] = 0.0
soil_df['Litter_N'] = 0.0
soil_df['Manure_C'] = 0.0
soil_df['Manure_N'] = 0.0
# soil_df['SoilFile'] = 'NELITCSE.soi'  #soil input file name for maizsim/glysim  <<<<<<==============
soil_df['SoilFile'] = 'NEMLTCRS.soi'  #soil input file name for maizsim/glysim  <<<<<<==============
soil_df['OM (%/100)'] = soil_df['OM'].astype(float)/100.0   #OM in fraction for Maizsim input
soil_df['CO2(ppm)'] = 400
soil_df['O2(ppm)'] = 20600

soil_df['sand_frac'] = soil_df['sand'].astype(float)/100.0
soil_df['silt_frac'] = soil_df['silt'].astype(float)/100.0
soil_df['clay_frac'] = soil_df['clay'].astype(float)/100.0

#set 3 decimal 
soil_df['Sand'] = round(soil_df['sand_frac'], 3)
soil_df['Silt'] = round(soil_df['silt_frac'], 3)
soil_df['Clay'] = round(soil_df['clay_frac'], 3)
soil_df['BD'] = round(soil_df['bd'].astype(float), 3)
#if soil_df['bd1'].values < 0.7:
    #soil_df['bd1'].values = 1.011

#===========check if TH1500, thm, Ks exists in the downloaded soil profile. If not, use default value (-1)
soil_df['TH33'] = round(soil_df['TH33'].astype(float), 3)
if 'TH1500' in soil_df.columns:  #Note: some locations might not have 'TH1500' (wsatiated_r)
    soil_df['TH1500'] = round(soil_df['TH1500'].astype(float), 3)  
else:
    soil_df['TH1500'] = -1

if 'thm' in soil_df.columns: #Note: some locations do not have 'thm' (wsatiated_r)
    soil_df['thm'] = round(soil_df['thm'].astype(float), 3)  
else:
    soil_df['thm'] = -1

if 'Ks' in soil_df.columns: #Note: some locations do not have 'thm' (wsatiated_r)
    soil_df['Ks'] = round(soil_df['Ks'].astype(float), 3)  
else:
    soil_df['Ks'] = -1

#soil_df['clay1']=round(soil_df['clay_frac'], 2)
# Select rows with max prcent
soil_df = soil_df[soil_df.prcent == soil_df.prcent.max()]

# Sort rows by depth
soil_df = soil_df.sort_values(by=['Bottom depth'])

# Check for rows with NaN values
soil_df_with_NaN = soil_df[soil_df.isnull().any(axis=1)]
depth = ", ".join(soil_df_with_NaN["Bottom depth"].astype(str))
if len(depth) > 0:
    #messageUser("Layers with the following depth " + depth + " were deleted.")
    soil_df = soil_df.dropna()

#=========== check if thm or TH1500 or ksat is valid. If not, put -1 (so as to estimate from ROSETTTA)
soil_df = soil_df[["soilFile", "Bottom depth", "Init Type", "OM (%/100)", "Humus_C", "Humus_N","Litter_C","Litter_N","Manure_C","Manure_N","NO3 (ppm)", "NH4", 
                "HNew", "Tmpr","CO2(ppm)","O2(ppm)", "Sand", "Silt", "Clay", "BD", "TH33", "TH1500", "thr", "ths", "tha", "thm", "Alfa", "n", "Ks", "Kk", "thk"]]

# Find the row index
row_index = soil_df[soil_df['TH33'] >= soil_df['thm']].index
if not row_index.empty:
    soil_df['thm'] = -1

row_index = soil_df[soil_df['thm'] == 0].index
if not row_index.empty:
    print("soil_df['thm'] == 0")
    soil_df['thm'] = -1
    os.system("pause")

row_index = soil_df[soil_df['Ks'] == 0].index
if not row_index.empty:
    soil_df['Ks'] = -1
    print("soil_df['Ks'] == 0")
    os.system("pause")

row_index = soil_df[soil_df['TH1500'] == 0].index
if not row_index.empty:
    soil_df['TH1500'] = -1
    print("soil_df['TH1500'] == 0]")
    os.system("pause")

if soil_df['BD'].iloc[0] < 0.7:
    print('check BD') 
    print(point) 
    print(soil_df)
    os.system("pause")
if soil_df['Clay'].iloc[0] < 0.04:
    print('check Clay') 
    print(point) 
    print(soil_df)
    os.system("pause")
if soil_df['Silt'].iloc[0] < 0.04:
    print('check Silt') 
    print(point) 
    print(soil_df)
    os.system("pause")
if soil_df['Sand'].iloc[0] < 0.04:
    print('check Sand') 
    print(point) 
    print(soil_df)
    os.system("pause")

#=========================================
#ADD OM decomposition parameters (*.noi) kh, kL, km etc from column AG => some default parameters....
soil_df['kh'] = 0.00007
soil_df['kL'] = 0.035
soil_df['km'] = 0.07
soil_df['kn'] = 0.2
soil_df['kd'] = 0.0000001
soil_df['fe'] = 0.6
soil_df['fh'] = 0.2
soil_df['r0'] = 10
soil_df['rL'] = 50
soil_df['rm'] = 10
soil_df['fa'] = 0.1
soil_df['nq'] = 8
soil_df['cs'] = 0.00001
#=========================================
# Rename one column
soil_df = soil_df.rename(columns={'thm': 'th'}) 
out_fname = os.path.join('.\\', point + '.csv')
soil_df.to_csv(out_fname, index=False)

