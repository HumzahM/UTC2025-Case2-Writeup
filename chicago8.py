#This was my final submission, plus some comments
#All of the work was done in allocate_portfolio(), rest was given

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#changed this line from what they gave
data = pd.read_csv('Case2.csv')

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.99, shuffle = False)

#They gave this class template to us
class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        self.running_price_paths = train_data.copy()
        self.tick_number = len(self.running_price_paths) - 1
        
    #All the work was done in here
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''
        self.running_price_paths = pd.concat([self.running_price_paths, pd.DataFrame([asset_prices])], ignore_index=True)
        self.tick_number += 1

        idx = self.tick_number % 10
        if idx == 0: #The return would sometimes stay the same direction, but usually switch. If it was completely 50/50 then optimal target == 0
            recent_data = self.running_price_paths.tail(2).copy()
            target = 0.33 #I tuned this after I tuned target_return. In theory I should've done both at the same time
            #expect the return sign to be opposite of the last one but otherwise difficult to predict - weigh equally
            w4 = -target if recent_data['Asset_4'].pct_change().iloc[-1] > 0 else target
            w5 = -target if recent_data['Asset_5'].pct_change().iloc[-1] > 0 else target
            w6 = -target if recent_data['Asset_6'].pct_change().iloc[-1] > 0 else target
            return np.array([0.0, 0.0, 0.0, w4, w5, w6]) 
        else: 
            assets = ['Asset_4', 'Asset_5', 'Asset_6']
            next_returns = [0, 0, 0]
            recent_data = self.running_price_paths[assets].tail(idx+1).copy()
            for i, asset in enumerate(assets):
                #in this case it is likely bouncing around near zero
                if recent_data[asset].pct_change().min() < 0 and recent_data[asset].pct_change().max() > 0:
                    value = 0
                else:
                    #Really, this should've taken advantage of the negative autocorrelation (the bouncing) pattern present
                    #and not just been the simple mean. However, the one with the highest mean generally had the least bouncing so it was also the least risky
                    value = recent_data[asset].pct_change().mean()
                next_returns[i] = value

            #Pull the return with the highest absolute value
            #In addition, the one with the highest mean usually had the least bouncing (based on how the data was being generated)
            max_idx = np.argmax(np.abs(next_returns))
            expected_return = abs(next_returns[max_idx])

            #This is something you'd obviously never do in real life (especially given the chance of a positive return was nearly 100%)
            #But "Capping" the return under the 50th percentile to reduce variance of the overall distribution worked quite well to increase the sharpe
            target_return = 0.0036
            if expected_return > target_return:
                weight = target_return / expected_return
                expected_return = target_return
            else:
                weight = 1

            #Zero on all assets except the one
            answer = np.zeros(6)
            answer[max_idx+3] = weight * np.sign(next_returns[max_idx])
            return answer

#They gave this function to us 
def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
print(sharpe)
#This should output 1.7306987660831863