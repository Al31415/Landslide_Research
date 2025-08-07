"""
SSURGO Data Collector Module
Collects soil data from USDA SSURGO database using SOAP API.
"""

import requests
import xmltodict
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings


class SSURGODataCollector:
    """
    Collect soil data from USDA SSURGO database.
    
    This module queries the SSURGO SOAP API to retrieve soil properties
    for given latitude/longitude coordinates.
    """

    def __init__(self, timeout: int = 60):
        """
        Initialize the SSURGO data collector.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.base_url = "https://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx"
        self.headers = {'content-type': 'text/xml'}

    def _build_soap_query(self, lat: float, lon: float) -> str:
        """
        Build SOAP query for SSURGO data.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            SOAP XML query string
        """
        lon_lat = f"{lon} {lat}"
        
        query = f"""<?xml version="1.0" encoding="utf-8"?>
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
                        FULL OUTER JOIN comonth cm  ON co.cokey=cm.comonthkey
                        FULL OUTER JOIN cosoilmoist csm  ON cm.comonthkey=csm.comonthkey
                        FULL OUTER JOIN chorizon ch ON co.cokey=ch.cokey
                        FULL OUTER JOIN chtexturegrp ctg ON ch.chkey=ctg.chkey
                        FULL OUTER JOIN chtexture ct ON ctg.chtgkey=ct.chtgkey
                        FULL OUTER JOIN copmgrp pmg ON co.cokey=pmg.cokey
                        FULL OUTER JOIN corestrictions rt ON co.cokey=rt.cokey
                        WHERE mu.mukey IN (SELECT * from SDA_Get_Mukey_from_intersection_with_WktWgs84('point({lon_lat})')) order by co.cokey, ch.chkey, prcent, deptht
            </sdm:Query>
          </sdm:RunQuery>
       </soap:Body>
    </soap:Envelope>"""
        
        return query

    def get_soil_data(self, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """
        Get soil data for a single coordinate pair.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            DataFrame with soil data or None if failed
        """
        try:
            # Build query
            body = self._build_soap_query(lat, lon)
            
            # Make request
            response = requests.post(
                self.base_url, 
                data=body, 
                headers=self.headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            my_dict = xmltodict.parse(response.content)
            
            # Extract data
            result = my_dict['soap:Envelope']['soap:Body']['RunQueryResponse']['RunQueryResult']
            
            if 'diffgr:diffgram' not in result:
                warnings.warn(f"No soil data found for coordinates ({lat}, {lon})")
                return None
            
            # Convert to DataFrame
            soil_df = pd.DataFrame.from_dict(
                result['diffgr:diffgram']['NewDataSet']['Table']
            )
            
            # Add coordinate information
            soil_df['latitude'] = lat
            soil_df['longitude'] = lon
            
            return soil_df
            
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Request failed for coordinates ({lat}, {lon}): {e}")
            return None
        except Exception as e:
            warnings.warn(f"Error processing soil data for coordinates ({lat}, {lon}): {e}")
            return None

    def get_soil_data_batch(self, coordinates: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        Get soil data for multiple coordinate pairs.
        
        Args:
            coordinates: List of (lat, lon) tuples
            
        Returns:
            DataFrame with soil data for all coordinates
        """
        frames = []
        
        for i, (lat, lon) in enumerate(coordinates):
            print(f"Processing coordinate {i+1}/{len(coordinates)}: ({lat:.5f}, {lon:.5f})")
            
            soil_df = self.get_soil_data(lat, lon)
            if soil_df is not None and not soil_df.empty:
                frames.append(soil_df)
            else:
                # Add empty row with coordinates
                empty_df = pd.DataFrame({
                    'latitude': [lat],
                    'longitude': [lon]
                })
                frames.append(empty_df)
        
        if frames:
            return pd.concat(frames, ignore_index=True)
        else:
            return pd.DataFrame()

    def extract_primary_soil_properties(self, soil_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract primary soil properties from SSURGO data.
        
        Args:
            soil_df: Raw SSURGO DataFrame
            
        Returns:
            DataFrame with primary soil properties
        """
        if soil_df.empty:
            return pd.DataFrame()
        
        # Define key soil properties to extract
        key_properties = [
            'cokey', 'wtenthbar_r', 'wthirdbar_r', 'wetting_front',
            'prcent', 'slope_r', 'hydgrp', 'runoff', 'erocl', 'pi_r',
            'slopelenusle_r', 'slope', 'hzname', 'deptht', 'depthb',
            'awc', 'clay', 'silt', 'partdensity', 'sand', 'OM',
            'bulk_density', 'th33', 'pH', 'sat_hidric_cond', 'bd',
            'latitude', 'longitude'
        ]
        
        # Filter columns that exist in the DataFrame
        available_cols = [col for col in key_properties if col in soil_df.columns]
        
        if not available_cols:
            warnings.warn("No expected soil properties found in data")
            return pd.DataFrame()
        
        # Extract primary properties
        primary_df = soil_df[available_cols].copy()
        
        # Convert numeric columns
        numeric_cols = [
            'wtenthbar_r', 'wthirdbar_r', 'wetting_front', 'prcent', 
            'slope_r', 'runoff', 'erocl', 'pi_r', 'slopelenusle_r', 
            'slope', 'deptht', 'depthb', 'awc', 'clay', 'silt', 
            'partdensity', 'sand', 'OM', 'bulk_density', 'th33', 
            'pH', 'sat_hidric_cond', 'bd'
        ]
        
        for col in numeric_cols:
            if col in primary_df.columns:
                primary_df[col] = pd.to_numeric(primary_df[col], errors='coerce')
        
        return primary_df

    def aggregate_soil_properties(self, soil_df: pd.DataFrame, 
                                aggregation_method: str = 'weighted_mean') -> pd.DataFrame:
        """
        Aggregate soil properties across multiple soil components.
        
        Args:
            soil_df: DataFrame with soil properties
            aggregation_method: Method for aggregation ('weighted_mean', 'mean', 'max')
            
        Returns:
            DataFrame with aggregated soil properties
        """
        if soil_df.empty:
            return pd.DataFrame()
        
        # Group by location
        grouped = soil_df.groupby(['latitude', 'longitude'])
        
        aggregated_data = []
        
        for (lat, lon), group in grouped:
            if group.empty:
                continue
            
            # Calculate component percentage for weighting
            if 'prcent' in group.columns:
                group['prcent'] = pd.to_numeric(group['prcent'], errors='coerce')
                total_pct = group['prcent'].sum()
                if total_pct > 0:
                    group['weight'] = group['prcent'] / total_pct
                else:
                    group['weight'] = 1.0 / len(group)
            else:
                group['weight'] = 1.0 / len(group)
            
            # Aggregate numeric properties
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude', 'weight']]
            
            aggregated_row = {'latitude': lat, 'longitude': lon}
            
            for col in numeric_cols:
                if aggregation_method == 'weighted_mean':
                    # Weighted mean
                    weighted_sum = (group[col] * group['weight']).sum()
                    aggregated_row[col] = weighted_sum
                elif aggregation_method == 'mean':
                    # Simple mean
                    aggregated_row[col] = group[col].mean()
                elif aggregation_method == 'max':
                    # Maximum value
                    aggregated_row[col] = group[col].max()
                else:
                    # Default to mean
                    aggregated_row[col] = group[col].mean()
            
            # For categorical properties, take the most common
            categorical_cols = ['hydgrp', 'hzname']
            for col in categorical_cols:
                if col in group.columns:
                    mode_value = group[col].mode()
                    aggregated_row[col] = mode_value.iloc[0] if not mode_value.empty else None
            
            aggregated_data.append(aggregated_row)
        
        return pd.DataFrame(aggregated_data)

    def collect_soil_data_for_dataset(self, df: pd.DataFrame,
                                    lat_col: str = 'Latitude',
                                    lon_col: str = 'Longitude',
                                    aggregate: bool = True) -> pd.DataFrame:
        """
        Collect soil data for all coordinates in a dataset.
        
        Args:
            df: DataFrame with latitude and longitude columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            aggregate: Whether to aggregate soil properties
            
        Returns:
            DataFrame with original data plus soil properties
        """
        # Get coordinates
        coordinates = list(zip(df[lat_col], df[lon_col]))
        
        # Collect soil data
        soil_df = self.get_soil_data_batch(coordinates)
        
        if soil_df.empty:
            print("No soil data collected")
            return df.copy()
        
        # Extract primary properties
        primary_df = self.extract_primary_soil_properties(soil_df)
        
        if primary_df.empty:
            print("No primary soil properties extracted")
            return df.copy()
        
        # Aggregate if requested
        if aggregate:
            primary_df = self.aggregate_soil_properties(primary_df)
        
        # Merge with original data
        result_df = df.copy()
        
        # Add soil properties (excluding lat/lon columns to avoid conflicts)
        soil_cols = [col for col in primary_df.columns if col not in [lat_col, lon_col]]
        
        for _, soil_row in primary_df.iterrows():
            # Find matching row in original data
            mask = ((result_df[lat_col] == soil_row['latitude']) & 
                   (result_df[lon_col] == soil_row['longitude']))
            
            if mask.any():
                for col in soil_cols:
                    result_df.loc[mask, f'soil_{col}'] = soil_row[col]
        
        return result_df


def collect_ssurgo_data_for_coordinates(coordinates: List[Tuple[float, float]]) -> pd.DataFrame:
    """
    Convenience function to collect SSURGO data for coordinates.
    
    Args:
        coordinates: List of (lat, lon) tuples
        
    Returns:
        DataFrame with soil properties
    """
    collector = SSURGODataCollector()
    return collector.get_soil_data_batch(coordinates)


if __name__ == "__main__":
    # Example usage
    collector = SSURGODataCollector()
    
    # Test with single coordinate
    lat, lon = 37.0, -120.0
    try:
        soil_data = collector.get_soil_data(lat, lon)
        if soil_data is not None and not soil_data.empty:
            print(f"Soil data for ({lat}, {lon}):")
            print(soil_data.head())
            
            # Extract primary properties
            primary_props = collector.extract_primary_soil_properties(soil_data)
            print(f"\nPrimary soil properties:")
            print(primary_props.head())
        else:
            print(f"No soil data found for ({lat}, {lon})")
    except Exception as e:
        print(f"Error collecting soil data: {e}")
    
    # Test batch collection
    coordinates = [(37.0, -120.0), (37.9, -121.2)]
    try:
        batch_data = collector.get_soil_data_batch(coordinates)
        if not batch_data.empty:
            print(f"\nBatch collection results:")
            print(batch_data.head())
            
            # Aggregate properties
            aggregated = collector.aggregate_soil_properties(batch_data)
            print(f"\nAggregated soil properties:")
            print(aggregated)
        else:
            print("No batch data collected")
    except Exception as e:
        print(f"Error in batch collection: {e}") 