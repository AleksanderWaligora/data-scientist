

def get_over_z_score_index(df,column,z_value=3):
  
    z_score = abs((df[column]-df[column].mean())/df[column].std()) > z_value
    over_z_score_indexes = z_score[z_score==True].index
    return list(over_z_score_indexes)

