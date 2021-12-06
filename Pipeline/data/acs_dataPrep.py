import json
import os
import urllib.request

import ohio
from tqdm import tqdm

API_KEY = ''
CENSUS_GOV = 'https://api.census.gov/data'


def query(url):
    """
    make query and get result
    :param url: url for API call
    :return: result of API call
    """
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read())
    return data


def get_geo_url(geo_list):
    """
    :param geo_list: list of (geo, geo_id) pair (smallest first)
                     e.g. [(block%20group, *), (state, 01), (county, 02), (track, *), ]
    :return: url for geo query
    """
    total_level = len(geo_list)
    url = ""
    for i in range(total_level):
        level, val = geo_list[i]
        if i == 0:
            url += "&for=%s:%s" % (level, val)
        else:
            url += "&in=%s:%s" % (level, val)
    return url


def url_builder(base_url, year=None, dataset=None, variables=None, geo=None, predicates=None, key=API_KEY):
    """
    build API query url for ACS data
    :param base_url: census_gov base url
    :param year: data year
    :param dataset: dataset name
    :param variables: list of variables ['var1' ,'var2', 'var3']
    :param geo: list of (geo, geo_id) pair
    :param predicates: list of predicates
    :param key: API Key
    :return: the composite url
    """
    if not year and not dataset:
        return base_url + '&key=%s' % API_KEY

    year_url = '/' + str(year)
    dataset_url = '/' + dataset
    variables_url = '?get=' + ','.join(variables)
    geo_url = get_geo_url(geo)
    key_url = '&key=%s' % API_KEY
    final = base_url + year_url + dataset_url + variables_url + geo_url + key_url
    return final


def build_state_index_map():
    """
    build a mapping between state name and state idx
    :return: state2index_map, index2state_map
    """
    state_index_mapping_url = url_builder("https://api.census.gov/data/2018/acs/acs1?&get=NAME&for=state:*")
    state_key = query(state_index_mapping_url)
    state2index, index2state = dict(), dict()
    for i in range(len(state_key[1:])):
        pair = state_key[1:][i]
        state2index[pair[0]] = pair[1]
        index2state[pair[1]] = pair[0]
    return state2index, index2state


def build_geo_list(level, state=None, county=None, tract=None, block=None):
    """
    build the geo query list as needed
    :param level: the smallest level you want to query
    :param state: state_id
    :param county: county_id
    :param tract:  tract_id
    :param block:  block_id
    :return: list of (geo, geo_id) pairs (smallest first)
    """
    if level == "county":
        return [('county', '*'),
                ('state', state)]
    elif level == "block":
        return [('block%20group', '*'),
                ('state', state),
                ('county', county),
                ('tract', '*')
                ]
    else:
        raise NotImplementedError


def build_state_county_map(state):
    """
    build county2index and index2county for the given state
    :param state: state id
    :return: mappings from county to county_id and vice versa for a state
    """
    county_index_mapping_url = url_builder(CENSUS_GOV,
                                           year=2018,
                                           dataset='acs/acs5',
                                           variables=['NAME'],
                                           geo=build_geo_list(level="county",
                                                              state=state))
    county_key = query(county_index_mapping_url)
    state_county2index = dict()
    state_index2county = dict()
    for i in range(len(county_key[1:])):
        pair = county_key[1:][i]
        county_name = pair[0].split(",")[0]
        state_county2index[county_name] = pair[-1]
        state_index2county[pair[-1]] = county_name
    return state_county2index, state_index2county


def block_iter_query_write(filename, variables, state):
    """
    write to file block-level data for variables given a state
    :param filename: filename to write
    :param variables: list of variables
    :param state: state id
    :return:
    """
    county_index_map, _ = build_state_county_map(state)

    first = True
    counties = list(county_index_map.keys())

    if os.path.exists(filename):
        f = open(filename, "w")
        f.close()

    for i in tqdm(range(len(counties))):
        county = counties[i]
        state_block = build_geo_list(level="block", state=state,
                                     county=county_index_map[county])
        block_url = url_builder(CENSUS_GOV,
                                year=2018,
                                dataset='acs/acs5',
                                variables=variables,
                                geo=state_block
                                )
        result = query(block_url)

        if not first:
            result = result[1:]
        else:
            first = False

        encoded_csv = ohio.encode_csv(result)
        with open(filename, "a") as file:
            file.write(encoded_csv)

    file.close()
    print("Write Complete!")


if __name__ == '__main__':
    state2index, index2state = build_state_index_map()

    variables = ['B01001_003E', 'B01001_004E', 'B01001_005E',
                 'B01001_018E', 'B01001_019E', 'B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E',
                 'B01001_024E',
                 'B01001_025E',
                 'B01001_027E', 'B01001_028E', 'B01001_029E', 'B01001_042E', 'B01001_043E', 'B01001_044E',
                 'B01001_045E', 'B01001_046E', 'B01001_047E', 'B01001_048E', 'B01001_049E',
                 'B11005_002E', 'B11006_002E',
                 'B19101_002E', 'B19101_003E', 'B19101_004E', 'B19101_005E', 'B19101_006E', 'B19101_007E',
                 'B19101_008E', 'B19101_009E', 'B19101_010E', 'B19101_011E',
                 'B27010_033E', 'B27010_050E'
                 ]

    block_iter_query_write("mike_acs.csv",
                           variables=variables,
                           state=state2index["Pennsylvania"])
