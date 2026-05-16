import numpy as np
import pandas as pd

# this method is development in progress and it is used to find the DZ and the SZ,
# eventually we will get to finding the implusive zones and swap zones with a little bit of math and dataframe manupulation
# Lets stick to finding the DZ and the SZ
# the combined dataframe which has rally base and drop base is used and applied on the dataframe to figure this out.
# if current ohlc is a rally base ,previous and next ohlc is rally then it is a RBR and proximal and distal gets assigned
# if current ohlc is a Drop base, previous and next ohlc is Drop then it is a RBR and proximal and distal gets assigned
# if there are multiple bases together then we combine them into one big base and then run the above two logics.


def identifyZones(DataSet):
    DataSet["group_id"] = DataSet.SubType.ne(DataSet.SubType.shift()).cumsum()
    nonBase_df = DataSet[(DataSet["SubType"] != "Base")].copy()

    Base_df = DataSet[(DataSet["SubType"] == "Base")].copy()
    Base_df["Close1"] = np.where(Base_df.Open < Base_df.Close, Base_df["Close"], Base_df["Open"])
    Base_df["Open"] = np.where(Base_df.Open > Base_df.Close, Base_df["Close"], Base_df["Open"])
    Base_df["Close"] = Base_df["Close1"]
    Base_df.drop("Close1", axis=1, inplace=True)

    Base_df["Open"] = Base_df.groupby(["SubType", "group_id"])["Open"].transform("min")
    Base_df["High"] = Base_df.groupby(["SubType", "group_id"])["High"].transform("max")
    Base_df["Low"] = Base_df.groupby(["SubType", "group_id"])["Low"].transform("min")
    Base_df["Close"] = Base_df.groupby(["SubType", "group_id"])["Close"].transform("max")
    Base_df = Base_df.drop_duplicates(subset=["Open", "High", "Close", "Low"])

    nonBase_df = pd.concat([nonBase_df, Base_df])
    nonBase_df = nonBase_df.sort_values(by="Date")

    nonBase_df["ZoneType"] = "helo"
    #####Below code figures out the proximal and distal for Rally Base Rally
    nonBase_df["Distal"] = 0
    nonBase_df["Proximal"] = 0
    fltr = (
        (nonBase_df["SubType"] == "Base")
        & (nonBase_df["SubType"].shift(1) == "Rally")
        & (nonBase_df["SubType"].shift(-1) == "Rally")
    )
    nonBase_df["Zone"] = np.where(fltr, "RBR", "0").copy()
    RBR_t = nonBase_df.isin(["RBR"]).any().any()

    if RBR_t:
        nonBase_df["Distal"] = np.where(
            nonBase_df["Zone"] == "RBR",
            nonBase_df["Low"].shift(-1).rolling(2, min_periods=0).min(),
            0,
        ).copy()
        nonBase_df["Proximal"] = np.where((nonBase_df["Zone"] == "RBR"), nonBase_df["Close"], 0)
        nonBase_df["cndl_bodycolor"] = np.where(
            nonBase_df["Zone"] == "RBR", "honeydew", nonBase_df["cndl_bodycolor"]
        )
        nonBase_df["cndl_llinecolor"] = np.where(
            nonBase_df["Zone"] == "RBR", "green", nonBase_df["cndl_llinecolor"]
        )
        nonBase_df["ZoneType"] = np.where(nonBase_df["Zone"] == "RBR", "DZ", "Z")
    # print(nonBase_df)
    #####Below code figures out the proximal and distal for Drop Base Drop

    fltr = (
        (nonBase_df["SubType"] == "Base")
        & (nonBase_df["SubType"].shift(1) == "Drop")
        & (nonBase_df["SubType"].shift(-1) == "Drop")
    )
    nonBase_df["Zone"] = np.where(fltr, "DBD", nonBase_df["Zone"]).copy()
    DBD_t = nonBase_df.isin(["DBD"]).any().any()

    if DBD_t:
        nonBase_df["Distal"] = np.where(
            (nonBase_df["Zone"] == "DBD"),
            nonBase_df["High"].shift(-1).rolling(2, min_periods=0).max(),
            nonBase_df["Distal"],
        ).copy()
        nonBase_df["Proximal"] = np.where(
            (nonBase_df["Zone"] == "DBD"), nonBase_df["Open"], nonBase_df["Proximal"]
        )
        nonBase_df["cndl_bodycolor"] = np.where(
            nonBase_df["Zone"] == "DBD", "mistyrose", nonBase_df["cndl_bodycolor"]
        )
        nonBase_df["cndl_llinecolor"] = np.where(
            nonBase_df["Zone"] == "DBD", "red", nonBase_df["cndl_llinecolor"]
        )

        nonBase_df["ZoneType"] = np.where(nonBase_df["Zone"] == "DBD", "SZ", nonBase_df["ZoneType"])

    #####Below code figures out the proximal and distal for Drop Base Rally
    fltr = (
        (nonBase_df["SubType"] == "Base")
        & (nonBase_df["SubType"].shift(1) == "Drop")
        & (nonBase_df["SubType"].shift(-1) == "Rally")
    )
    nonBase_df["Zone"] = np.where(fltr, "DBR", nonBase_df["Zone"])
    DBR_t = nonBase_df.isin(["DBR"]).any().any()

    if DBR_t:
        nonBase_df["Proximal"] = np.where(
            (nonBase_df["Zone"] == "DBR"), nonBase_df["Close"], nonBase_df["Proximal"]
        )
        # nonBase_df['Distal']=np.where((nonBase_df['Zone']=='DBR'),nonBase_df['Low'],nonBase_df['Distal'])
        nonBase_df["Distal"] = np.where(
            (nonBase_df["Zone"] == "DBR"),
            nonBase_df["Low"].shift(-1).rolling(3, min_periods=0).min(),
            nonBase_df["Distal"],
        )
        nonBase_df["cndl_bodycolor"] = np.where(
            nonBase_df["Zone"] == "DBR", "honeydew", nonBase_df["cndl_bodycolor"]
        )
        nonBase_df["cndl_llinecolor"] = np.where(
            nonBase_df["Zone"] == "DBR", "green", nonBase_df["cndl_llinecolor"]
        )
        nonBase_df["ZoneType"] = np.where(nonBase_df["Zone"] == "DBR", "DZ", nonBase_df["ZoneType"])

    #####Below code figures out the proximal and distal for Rally Base Drop
    fltr = (
        (nonBase_df["SubType"] == "Base")
        & (nonBase_df["SubType"].shift(1) == "Rally")
        & (nonBase_df["SubType"].shift(-1) == "Drop")
    )
    nonBase_df["Zone"] = np.where(fltr, "RBD", nonBase_df["Zone"])
    RBD_t = nonBase_df.isin(["RBD"]).any().any()

    if RBD_t:
        nonBase_df["Proximal"] = np.where(
            nonBase_df["Zone"] == "RBD", nonBase_df["Open"], nonBase_df["Proximal"]
        )
        nonBase_df["Distal"] = np.where(
            (nonBase_df["Zone"] == "RBD"),
            nonBase_df["High"].shift(-1).rolling(3, min_periods=0).max(),
            nonBase_df["Distal"],
        )
        nonBase_df["cndl_bodycolor"] = np.where(
            nonBase_df["Zone"] == "RBD", "mistyrose", nonBase_df["cndl_bodycolor"]
        )
        nonBase_df["cndl_llinecolor"] = np.where(
            nonBase_df["Zone"] == "RBD", "red", nonBase_df["cndl_llinecolor"]
        )

    nonBase_df["ZoneType"] = np.where(nonBase_df["Zone"] == "RBD", "SZ", nonBase_df["ZoneType"])
    # rint("nonBase_df")

    # print(nonBase_df)

    return nonBase_df
