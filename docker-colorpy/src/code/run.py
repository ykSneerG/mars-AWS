import json
from files.cgatsCreator import CgatsCreator
from files.cgats import Cgats
import os
from files.jsondata import JsonDataSorter


def clear_console():
    '''
    Clear the console screen
    '''
    os.system('cls' if os.name == 'nt' else 'clear')


def run_CgatsToJson(file_path='test/testdata/CMYK_Lin_SNM.txt'):
    '''
    Run the creator
    '''
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]

    cg = Cgats(lines)
    # res = cg.row_id(2)
    # print(cg.head.to_json())
    # print(cg.row_all())
    # print(cg.row_random())
    # print(cg.dcs_order([1, 0, 2, 3]))
    # print(cg.dcs_order([0, 2]))
    print(cg.dcs_order([0, 1, 2, 4]))


def run_JsonToCgats(file_path='test/testdata/CMYK_Lin_LAB.json'):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    cg = CgatsCreator()
    cg.set_json(json_data)
    res = cg.get_txt()
    print(res)


def run_JsonDataSorter(file_path='test/testdata/CMYK_Lin_LAB.json'):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    sorter = JsonDataSorter(json_data['data'])
    res = sorter.sort_visual()

    print(json.dumps(res))


def run_CgatsSortCgats(file_path='test/testdata/FOGRA39_MW3_Subset.txt', order_type=JsonDataSorter.OrderType.RANDOM):
    # 1. read TXT to list[str]
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]

    # 2. Convert Cgats to Json
    cg = Cgats(lines)

    # 3. Sort Json
    sorter = JsonDataSorter(cg.row_all(), cg.get_type_pcs())
    # cg.set_data(sorter.sort_random())
    '''
    if (order_type == JsonDataSorter.OrderType.LIGHTNESS or
            order_type == JsonDataSorter.OrderType.CHROMA):
        sorter.set_pcs_type(cg.get_type_pcs())
    '''
    cg.set_data(sorter.sort(order_type))

    # 4. Convert Json to Cgats
    cge = CgatsCreator()
    cge.set_json(cg.to_json())
    res = cge.get_txt()

    # 5. Output the result
    print(res)


# execute only if run as a script
if __name__ == "__main__":
    # clear_console()
    # run_CgatsToJson('test/testdata/CMYK_Lin_SNM.txt')
    # run_CgatsToJson('test/testdata/CMYK_Lin_LAB_mini.txt')
    # run_CgatsToJson('test/testdata/CMYK_Lin_XYZ.txt')
    # run_CgatsToJson('test/testdata/FOGRA39_MW3_Subset.txt')

    # run_JsonToCgats('test/testdata/CMYK_Lin_SNM.json')
    # run_JsonToCgats('test/testdata/CMYK_Lin_LAB.json')
    # run_JsonToCgats('test/testdata/test.json')

    # run_JsonDataSorter('test/testdata/FOGRA39_MW3_Subset.json')

    # run_JsonDataSorter(file_path='test/testdata/FOGRA39_MW3_Subset.json')

    run_CgatsSortCgats(file_path='test/testdata/FOGRA39_MW3_Subset.txt',
                       order_type=JsonDataSorter.OrderType.CHROMA)
    '''

    run_CgatsSortCgats(file_path='test/testdata/CMYK_Lin_SNM.txt',
                       order_type=JsonDataSorter.OrderType.LIGHTNESS)
    '''
