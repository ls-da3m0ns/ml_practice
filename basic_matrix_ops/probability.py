import numpy as np

def factorial(k):
    if k==1:
        return 1 
    elif k==2:
        return 2 
    else:
        return k*factorial(k-1)



def poisson_prob(lambda_ , num_events):
    """Calculates probability of observing num_events if rate of events is lambda_
        prob = lambda^num_events * (e^(-lambda)) / num_events!    
    """

    return (lambda_**num_events) * ( np.e**(-1 * lambda_) / factorial(num_events))


def binomial_prob(total_trials, succesfull_trials, prob_success):
    """
    Calculates probability of exactly succesfull_trials num of success in total_trials 
    using binomial distribution
        prob = nCr*p^r*(1-p)^(n-r)
        nCr = n!/(n-r)!(r!)
    """
    return (
        factorial(total_trials) \
            * prob_success**succesfull_trials \
            * (1-prob_success) ** (total_trials - succesfull_trials) 
            ) / (
                factorial(succesfull_trials) * factorial(total_trials - succesfull_trials)
            )

def normal_prob(mean, std, x):
    """
    Calculates probabilty of x given mean and std of Normal Distribution
     prob = (1/(2* pi * std^2)^1/2) * e ^ (-1 * (x-u)^2 / 2*std^2) 
    """
    return ((2 * np.pi * std**2) **(-1/2)) * (np.e**( (-1 * (x-mean)**2) / (2*std**2) ))


def sort_series(series):
    """
    sorts numeric series using merge sort algo
    """
    if len(series) <= 1:
        return series
    
    m= int(len(series) / 2)

    sorted_l = sort_series(series[:m])
    sorted_r = sort_series(series[m:])

    merged = []
    i,j = 0,0
    while i<len(sorted_l) and j < len(sorted_r):
        if sorted_l[i] <= sorted_r[j]:
            merged.append(sorted_l[i])
            i+=1
        else:
            merged.append(sorted_r[j])
            j+=1
    if i< len(sorted_l) : merged.extend(sorted_l[i:])
    elif j< len(sorted_r) : merged.extend(sorted_r[j:])

    return merged

def freq_counter(series):
    counter = {}
    for i in series:
        counter[i] = counter.get(i,0) + 1
    return counter 

def percentile(sorted_series, x):
    num_elements = len(sorted_series)
    position = (num_elements -1) * (x/100)
    tol = 0.0001
    lower = sorted_series[ max( 
        int(position),
        0 
    ) ]
    upper = sorted_series[ min( 
        int( position + (1 if ( position - int(position) ) > tol else 0)),
        num_elements -1 
    ) ]

    width = upper - lower 
    percent = (max( 
        int(position),
        0 
    ) / position )if position != 0 else 0

    return round( lower + width * percent, 5) 

def descriptive_stats(series):
    """
    Computes varies descriptive stats on a series of data
        Mean
        Median
        Mode 
        Variance
        Standard Deviation
        nth percentile 
        IQR
    """
    sorted_series = sort_series(series)
    freq_table = freq_counter(sorted_series)
    num_elems = len(sorted_series)

    mean = sum(sorted_series) / num_elems
    mode = sorted(freq_table.items(), key=lambda x: x[1])[-1][0]
    variance = sum( map(lambda x: (x-mean)**2 , sorted_series)) / num_elems
    std = variance**(1/2)
    median = percentile(sorted_series, 50)
    p25 = percentile(sorted_series, 25)
    p75 = percentile(sorted_series, 75)
    IQR = p75 - p25 
    return {
        "mean" : mean ,
        "median" : median,
        "variance" : variance,
        "std" : std,
        "p25" : p25,
        "p75" : p75,
        "IQR" : IQR
    } 


if __name__ == '__main__':
    num_events = 3 
    lamb = 5 

    print(
        "Poission Prob",
        poisson_prob(5,3)
    )

    n,r,p = 6,2,0.5
    print(
        "Binomila prob",
        binomial_prob(n,r,p)
    )

    x,mean,std = 16,15,2.04
    print(
        "Normal Prob",
        normal_prob(mean, std, x)
    )

    series = [10, 20, 30, 40, 50]
    print(sort_series(series) )

    print(descriptive_stats(series ))