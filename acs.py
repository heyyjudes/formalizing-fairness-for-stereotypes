#adult 
adult_race_grouping=  {
        1.0:1.0,
        2.0:2.0,
        3.0:3.0,
        4.0:3.0, # mapping "American Indian Alone" to other
        5.0:3.0, # mapping "Alaska Naive Alone" to other
        6.0:6.0, 
        7.0:3.0, # mapping "Native Hawaiian Alone" to other
        8.0:3.0, # mapping "Some Other Race alone" to other
        9.0:3.0, # mapping "Two or More Races" to other
    }

ACSIncome_categories = {
    "COW": {
        1.0: (
            "For-profit"
        ),
        2.0: (
            "Not-for-profit"
        ),
        3.0: "Local gov",
        4.0: "State gov",
        5.0: "Federal gov",
        6.0: (
            "Self-employed not incorporated"
        ),
        7.0: (
            "Self-employed incorporated"
        ),
        8.0: "Working without pay",
        9.0: "Unemployed",
    },

    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: "American Indian or Alaska Native, not specified",
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}


        
ACSPublicCoverage_categories = {
    #disability
    "DIS": {1.0: "With a disability", 2.0: "No disability"},
    # Employment status of parents
    "ESP":{
        0.0: "Other",
        1.0: "Two parents: Both Emp",
        2.0: "Two parents: Father Emp",
        3.0: "Two parents: Mother Emp",
        4.0: "Two parents: Unemp",
        5.0: "Father Emp",
        6.0: "Father Unemp",
        7.0: "Mother Emp",
        8.0: "Mother Unemp",
    },
    # citizenship status
    "CIT":{
        1.0: "US Born",
        2.0: "PR Born",
        3.0: "US Parents",
        4.0: "Naturalization",
        5.0: "Non-citizen",
    },
    #Mobility status (lived here 1 year ago)
    "MIG":{
        1.0: "Yes, same house (nonmovers)",
        2.0: "No, outside US and Puerto Rico",
        3.0: "No, different house in US or Puerto Rico",
    },
    # Military service

    "MIL":{
        0.0: "N/A",
        1.0: "Active duty",
        2.0: "Past",
        3.0: "Reseves",
        4.0: "Never",
    },
    # Ancestry Recode: 
    "ANC":{
        1.0: "Single",
        2.0: "Multiple",
        3.0: "Unclassified",
        4.0: "Not reported",
    },
    # nativity
    "NATIVITY":{
        1.0: "Native",
        2.0: "Foreign born",
    },
    # Hearing Difficulty
    "DEAR":{
        1.0: "Yes",
        2.0: "No",
    },
    # Vision Difficulty
    "DEYE":{"Yes": 1.0, "No": 2.0},
    # Employmnet Status Recode
    "ESR":{0.0: "N/A (less than 16 years old)",
        1.0: "Civilian employed, at work",
        2.0: "Civilian employed, with a job but not at work",
        3.0: "Unemployed",
        4.0: "Armed forces, at work",
        5.0: "Armed forces, with a job but not at work",
        6.0: "Not in labor force",
        },
# Gave Birth to child within the past 12 months
    "FER":{
        0.0: "N/A (less than 15 years old)",
        1.0: "Yes",
        2.0: "No",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: "American Indian or Alaska Native, not specified",
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}

ACSEmployment_categories = {
    #disability
    "DIS": {1.0: "With a disability", 2.0: "No disability"},
    # Employment status of parents
    "ESP":{
        0.0: "Other",
        1.0: "Two parents: Both Emp",
        2.0: "Two parents: Father Emp",
        3.0: "Two parents: Mother Emp",
        4.0: "Two parents: Unemp",
        5.0: "Father Emp",
        6.0: "Father Unemp",
        7.0: "Mother Emp",
        8.0: "Mother Unemp",
    },
    # citizenship status
    "CIT":{
        1.0: "US Born",
        2.0: "PR Born",
        3.0: "US Parents",
        4.0: "Naturalization",
        5.0: "Non-citizen",
    },
    #Mobility status (lived here 1 year ago)
    "MIG":{
        1.0: "Yes",
        2.0: "No, Inter",
        3.0: "No, Domestic",
    },
    # Military service

    "MIL":{
        0.0: "N/A",
        1.0: "Active duty",
        2.0: "Past",
        3.0: "Reseves",
        4.0: "Never",
    },
    # Ancestry Recode: 
    "ANC":{
        1.0: "Single",
        2.0: "Multiple",
        3.0: "Unclassified",
        4.0: "Not reported",
    },
    # nativity
    "NATIVITY":{
        1.0: "Native",
        2.0: "Foreign born",
    },
    # Hearing Difficulty
    "DEAR":{
        1.0: "Yes",
        2.0: "No",
    },
    # Vision Difficulty
    "DEYE":{"Yes": 1.0, "No": 2.0},
    # Cognitive difficulty
    "DREM":{0.0: "N/A (less than 5 years old)",
        1.0: "Yes",
        2.0: "No",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: "American Indian or Alaska Native, not specified",
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}