class CgatsConstants:

    UNSET = "unset"

    # ----- STRING KEYS -----
    ID_TPLS = ("ID", ['SAMPLE_ID'])
    NAME_TPLS = ("NAME", ['SAMPLE_NAME'])
    QTY_TPLS = ("QTY", ['SAMPLE_QTY'])

    # ----- DCS TPLS -----
    CMY_TPLS = ("CMY", ['CMYK_C', 'CMYK_M', 'CMYK_Y'])
    CMYK_TPLS = ("CMYK", ['CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K'])
    RGB_TPLS = ("RGB", ['RGB_R', 'RGB_G', 'RGB_B'])

    # 9CLR_1
    CLR1_TPLS = ("CLR1", ['1CLR_1'])
    CLR2_TPLS = ("CLR2", ['2CLR_1', '2CLR_2'])
    CLR3_TPLS = ("CLR3", ['3CLR_1', '3CLR_2', '3CLR_3'])
    CLR4_TPLS = ("CLR4", ['4CLR_1', '4CLR_2', '4CLR_3', '4CLR_4'])
    CLR5_TPLS = ("CLR5", ['5CLR_1', '5CLR_2', '5CLR_3', '5CLR_4', '5CLR_5'])
    CLR6_TPLS = ("CLR6", ['6CLR_1', '6CLR_2', '6CLR_3', '6CLR_4', '6CLR_5', '6CLR_6'])
    CLR7_TPLS = ("CLR7", ['7CLR_1', '7CLR_2', '7CLR_3', '7CLR_4', '7CLR_5', '7CLR_6', '7CLR_7'])
    CLR8_TPLS = ("CLR8", ['8CLR_1', '8CLR_2', '8CLR_3', '8CLR_4', '8CLR_5', '8CLR_6', '8CLR_7', '8CLR_8'])
    CLR9_TPLS = ("CLR9", ['9CLR_1', '9CLR_2', '9CLR_3', '9CLR_4', '9CLR_5', '9CLR_6', '9CLR_7', '9CLR_8', '9CLR_9'])
    CLR10_TPLS = ("CLR10", ['10CLR_1', '10CLR_2', '10CLR_3', '10CLR_4', '10CLR_5', '10CLR_6', '10CLR_7', '10CLR_8', '10CLR_9', '10CLR_10'])
    CLR11_TPLS = ("CLR11", ['11CLR_1', '11CLR_2', '11CLR_3', '11CLR_4', '11CLR_5', '11CLR_6', '11CLR_7', '11CLR_8', '11CLR_9', '11CLR_10', '11CLR_11'])
    CLR12_TPLS = ("CLR12", ['12CLR_1', '12CLR_2', '12CLR_3', '12CLR_4', '12CLR_5', '12CLR_6', '12CLR_7', '12CLR_8', '12CLR_9', '12CLR_10', '12CLR_11',
                            '12CLR_12'])
    CLR13_TPLS = ("CLR13", ['13CLR_1', '13CLR_2', '13CLR_3', '13CLR_4', '13CLR_5', '13CLR_6', '13CLR_7', '13CLR_8', '13CLR_9', '13CLR_10', '13CLR_11',
                            '13CLR_12', '13CLR_13'])

    # PC9_1
    PC1_TPLS = ("CLR1", ['PC1_1'])
    PC2_TPLS = ("CLR2", ['PC2_1', 'PC2_2'])
    PC3_TPLS = ("CLR3", ['PC3_1', 'PC3_2', 'PC3_3'])
    PC4_TPLS = ("CLR4", ['PC4_1', 'PC4_2', 'PC4_3', 'PC4_4'])
    PC5_TPLS = ("CLR5", ['PC5_1', 'PC5_2', 'PC5_3', 'PC5_4', 'PC5_5'])
    PC6_TPLS = ("CLR6", ['PC6_1', 'PC6_2', 'PC6_3', 'PC6_4', 'PC6_5', 'PC6_6'])
    PC7_TPLS = ("CLR7", ['PC7_1', 'PC7_2', 'PC7_3', 'PC7_4', 'PC7_5', 'PC7_6', 'PC7_7'])
    PC8_TPLS = ("CLR8", ['PC8_1', 'PC8_2', 'PC8_3', 'PC8_4', 'PC8_5', 'PC8_6', 'PC8_7', 'PC8_8'])
    PC9_TPLS = ("CLR9", ['PC9_1', 'PC9_2', 'PC9_3', 'PC9_4', 'PC9_5', 'PC9_6', 'PC9_7', 'PC9_8', 'PC9_9'])
    PC10_TPLS = ("CLR10", ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5', 'PCA_6', 'PCA_7', 'PCA_8', 'PCA_9', 'PCA_A'])
    PC11_TPLS = ("CLR11", ['PCB_1', 'PCB_2', 'PCB_3', 'PCB_4', 'PCB_5', 'PCB_6', 'PCB_7', 'PCB_8', 'PCB_9', 'PCB_A', 'PCB_B'])
    PC12_TPLS = ("CLR12", ['PCC_1', 'PCC_2', 'PCC_3', 'PCC_4', 'PCC_5', 'PCC_6', 'PCC_7', 'PCC_8', 'PCC_9', 'PCC_A', 'PCC_B', 'PCC_C'])
    PC13_TPLS = ("CLR13", ['PCD_1', 'PCD_2', 'PCD_3', 'PCD_4', 'PCD_5', 'PCD_6', 'PCD_7', 'PCD_8', 'PCD_9', 'PCD_A', 'PCD_B', 'PCD_C', 'PCD_D'])

    @staticmethod
    def get_list_dcs():
        return [CgatsConstants.CMYK_TPLS,
                CgatsConstants.CMY_TPLS,
                CgatsConstants.RGB_TPLS,
                CgatsConstants.CLR1_TPLS,
                CgatsConstants.CLR2_TPLS,
                CgatsConstants.CLR3_TPLS,
                CgatsConstants.CLR4_TPLS,
                CgatsConstants.CLR5_TPLS,
                CgatsConstants.CLR6_TPLS,
                CgatsConstants.CLR7_TPLS,
                CgatsConstants.CLR8_TPLS,
                CgatsConstants.CLR9_TPLS,
                CgatsConstants.CLR10_TPLS,
                CgatsConstants.CLR11_TPLS,
                CgatsConstants.CLR12_TPLS,
                CgatsConstants.CLR13_TPLS,
                CgatsConstants.PC1_TPLS,
                CgatsConstants.PC2_TPLS,
                CgatsConstants.PC3_TPLS,
                CgatsConstants.PC4_TPLS,
                CgatsConstants.PC5_TPLS,
                CgatsConstants.PC6_TPLS,
                CgatsConstants.CLR7_TPLS,
                CgatsConstants.PC8_TPLS,
                CgatsConstants.PC9_TPLS,
                CgatsConstants.PC10_TPLS,
                CgatsConstants.PC11_TPLS,
                CgatsConstants.PC12_TPLS,
                CgatsConstants.PC13_TPLS]

    # ----- PCS TPLS -----
    SNM_TPLS = ("SNM", [
                'SPECTRAL_NM_380', 'SPECTRAL_NM_390', 'SPECTRAL_NM_400',
                'SPECTRAL_NM_410', 'SPECTRAL_NM_420', 'SPECTRAL_NM_430', 'SPECTRAL_NM_440', 'SPECTRAL_NM_450',
                'SPECTRAL_NM_460', 'SPECTRAL_NM_470', 'SPECTRAL_NM_480', 'SPECTRAL_NM_490', 'SPECTRAL_NM_500',
                'SPECTRAL_NM_510', 'SPECTRAL_NM_520', 'SPECTRAL_NM_530', 'SPECTRAL_NM_540', 'SPECTRAL_NM_550',
                'SPECTRAL_NM_560', 'SPECTRAL_NM_570', 'SPECTRAL_NM_580', 'SPECTRAL_NM_590', 'SPECTRAL_NM_600',
                'SPECTRAL_NM_610', 'SPECTRAL_NM_620', 'SPECTRAL_NM_630', 'SPECTRAL_NM_640', 'SPECTRAL_NM_650',
                'SPECTRAL_NM_660', 'SPECTRAL_NM_670', 'SPECTRAL_NM_680', 'SPECTRAL_NM_690', 'SPECTRAL_NM_700',
                'SPECTRAL_NM_710', 'SPECTRAL_NM_720', 'SPECTRAL_NM_730'])
    LAB_TPLS = ("LAB", ['LAB_L', 'LAB_A', 'LAB_B'])
    XYZ_TPLS = ("XYZ", ['XYZ_X', 'XYZ_Y', 'XYZ_Z'])
    LCH_TPLS = ("LCH", ['LCH_L', 'LCH_C', 'LCH_H'])

    @staticmethod
    def get_list_pcs():
        return [CgatsConstants.SNM_TPLS,
                CgatsConstants.LAB_TPLS,
                CgatsConstants.XYZ_TPLS,
                CgatsConstants.LCH_TPLS]

    # ----- staticmethod ----- staticmethod ----- staticmethod ----- staticmethod ----- staticmethod -----

    @staticmethod
    def find_tuple_by_string(lst, search_string):
        for tup in lst:
            if tup[0] == search_string:
                return tup
        return None

    # ----- STRING KEYWORDS -----
    BEGIN_DATA = 'BEGIN_DATA'
    END_DATA = 'END_DATA'

    BEGIN_DATA_FORMAT = 'BEGIN_DATA_FORMAT'
    END_DATA_FORMAT = 'END_DATA_FORMAT'

    LGOROWLENGTH = ['LGOROWLENGTH']
    '''
    '''
    ORIGINATOR = ['ORIGINATOR']
    '''
    Identifies the specific system, organization or individual that created the data file.
    '''
    FILE_DESCRIPTOR = ['FILE_DESCRIPTOR', 'DESCRIPTOR']
    CREATED = ['CREATED']
    '''
    Indicates the creation date of the data file. The form for this date is CCYY-MM-DDThh:mm:ss[Z | +/-hh:mm].
    '''
    INSTRUMENTATION = ['INSTRUMENTATION']
    MEASUREMENT_GEOMETRY = ['MEASUREMENT_GEOMETRY']
    MEASUREMENT_SOURCE = ['MEASUREMENT_SOURCE']
    FILTER = ['FILTER']
    POLARIZATION = ['POLARIZATION']
    WEIGHTING_FUNCTION = ['WEIGHTING_FUNCTION']
    COMPUTATIONAL_PARAMETER = ['COMPUTATIONAL_PARAMETER']
    SAMPLE_BACKING = ['SAMPLE_BACKING']
    MANUFACTURER = ['MANUFACTURER']
    MATERIAL = ['MATERIAL']
    TARGET_TYPE = ['TARGET_TYPE']
    COLORANT = ['COLORANT']
    PROD_DATE = ['PROD_DATE']
    PRINT_CONDITIONS = ['PRINT_CONDITIONS']
    SERIAL = ['SERIAL']
    PROCESSCOLOR_ID = ['PROCESSCOLOR_ID']
    SPOT_ID = ['SPOT_ID']

    KEYWORD = ['KEYWORD']

    '''
    CGATS.17
    '''
    # ----- CAT -----
    '''
    dcsCategory = {
        "WP": 'WP' # WHITEPOINT
        "P": 'P'   # PRIME
        "F": 'F'   # Full
        "L": 'L'   # LIMIT
        "U1": 'U1' # Lighter than a given Color.
        # >>>>>>>>>>>>>>>>>>>>>>>
        # Not directly a DcsCategory,
        # should be refactored in the future
        # >>>>>>>>>>>>>>>>>>>>>>>
        # COLOR
        "C": 'C'
    }
    '''
