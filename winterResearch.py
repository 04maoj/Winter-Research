import json
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math
import statistics
import seaborn as sns
import pandas
import peakutils
from dtaidistance import dtw
import tslearn.metrics


def myDTW(s1, s2):
    n = len(s1)
    m = len(s2)
    opt = [[10000 for i in range(m + 1)] for j in range(n + 1)]
    opt[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            opt[i][j] = cost + min(opt[i - 1][j], opt[i][j - 1], opt[i - 1][j - 1])

    return opt[n][m]

## Method used for reading Json File, output is an array of byte, an array of timestamp,
## an array of timestamp indicating when we called the application.
def readJson(data_file,time_stamp_file):
    weather =[]
    time_stamp = []
    for line in open(data_file,'r'):
        weather.append(json.loads(line))
    if time_stamp_file is not None:
        for line in open(time_stamp_file,'r'):
            time_stamp.append(int(float(line.strip())))
        init_2 = time_stamp[0]
    else:
        init_2 = weather[0]['ts'];
    dic_2 = {}
    ts_2 = []
    maxByte = 0;
    for js in weather:
        if not dic_2.__contains__(js['ts']-init_2):
            dic_2[js['ts']- init_2] = js['total_byte_count']
        else:
            dic_2[js['ts']-init_2] += js['total_byte_count']
        if maxByte < js['total_byte_count'] and js['ts']>=init_2 :
            maxByte = max(maxByte, js['total_byte_count'])
    byte_2=[]
    for i in range (0,weather[len(weather)-1]['ts']-init_2,1):
        ts_2.append(i)
        if dic_2.__contains__(i):
            byte_2.append(dic_2[i])
        else:
            byte_2.append(0)
    indicators_x = []
    indicators_y = []
    if time_stamp_file is not None:
        for i in range(0,len(time_stamp)):
            indicators_x.append(int(time_stamp[i])-init_2)
            indicators_y.append(250000)
    return byte_2, ts_2, indicators_x,indicators_y,250000

##Function for plotting subset of the Json File graphs one can specific the subset by changing the variable time
def plotSubGraph(time,day_i,audio,time_s):
    color = ["b","y","r"]
    question =["What is the weather today","What time is it","Distance from Paris"]
    fig = plt.figure(time)
    fig.ylim(top=850000)
    fig.ylim(bottom=0)
    plt.title("Web Traffic for set "+str(time))
    plt.xlabel("Time")
    plt.ylabel("Total Byte")
    ax = fig.add_subplot(111)
    day = day_i
    fileName = "Audio_Data/SUM/" + audio+".json"
    time_stamp_file = "Audio_Data/SUM/ts_"+ time_s+".txt"
    byte,ts,indicators_x,indicators_y= readJson(fileName,time_stamp_file)
    start = time * 8 * 3
    end = ((time+1) * 8 * 3)
    start_p = 0;
    for u in range(0,(len(byte))):
    if ts[u] == indicators_x[start]:
        start_p = u
    if ts[u] == indicators_x[end-1]:
        end_p = u
    byte = byte[start_p:end_p + 10]
    ts = ts[start_p:end_p + 10]
    indicators_x = indicators_x[start:end]
    indicators_y = indicators_y[start:end]

    for p in range(0,3):
        subset_x = []
        subset_y = []
        count = 0
        for q in range (0, len(indicators_y)):
            if count == p:
                subset_y.append(indicators_y[q])
                subset_x.append(indicators_x[q])
            count +=1
            if count == 3:
                count = 0
        plt.plot(subset_x,subset_y,'*',color=color[p], label=question[p])
    ax.plot(ts, byte, '.-', color=color[i-1], label=("Day " + str(day)+ " set " + str(j+1)))
    plt.legend(loc="best")

##Plotting the whole graph for Json File
def plotWholeGraph(day):
    color = ["b","y","r"]
    question =["What is the weather today","What time is it","Distance from Paris"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    maximum = -1
    max_index = -1;
    for i in range(1, 2):
        fileName = "Audio_Data/SUM/day_" + str(day) + ".json"
        time_stamp_file = "Audio_Data/SUM/ts_" + str(day) + ".txt"
        # time_stamp_file = None
        byte, ts, indicators_x, indicators_y = readJson(fileName, time_stamp_file)
        ax.plot(ts, byte, '.-', color=color[i - 1], label=("Day " + str(day) + " set " + str(j + 1)))

##Used to return standard deviation, mean, list of byte for a certain question, the average byte 'block_range' seconds
## after the initial call
def find_Stats(question_number,byte, ts, indicators_x, in_block = False, block_range = 10,all = False):
    count = 0
    byte_to_count =[]
    if all == False:
        for j in range(0, len(indicators_x)):
            if count == question_number:
                byte_to_count.append(indicators_x[j])
            count +=1
            if count == 3:
                count = 0
    else:
        byte_to_count = indicators_x
    j = 0
    out = []
    first_ten_second = [[]  for i in range(block_range)]
    if(in_block == False):
        for i in range(0,len(ts)):
            if(j >= len(byte_to_count)):
                break
            if(ts[i] >= byte_to_count[j]):
                max_n = 0
                temp = 0
                while(i< len(ts) and ts[i] < byte_to_count[j] + block_range):
                    max_n += byte[i]
                    first_ten_second[temp].append(byte[i])
                    temp+=1
                    i+=1
                out.append(max_n)
                j += 1
    else:
        for i in range(0,len(ts)):
            if(j >= len(byte_to_count)):
                break
            if(ts[i] >= byte_to_count[j] and ts[i] <=  byte_to_count[j]+block_range):
                out.append(byte[i])
                j += 1
    generatedLineForDTW  = []
    for j in range(0, block_range):
        mu,sigma = filter_outlier(first_ten_second[j])
        generatedLineForDTW.append(mu)
    return statistics.stdev(out),statistics.mean(out),out,generatedLineForDTW

## Function usted to return the mean and standard deviation of a sequence without outlier.
def filter_outlier(out):
    Q1 = np.percentile(out, 25)
    Q3 = np.percentile(out, 75)
    IQR = Q3 - Q1
    new_out = []
    for a in out:
        if not (a < Q1 - 1.5 * IQR) and not (a > Q3 + 1.5 * IQR):
            new_out.append(a)

    mu = statistics.mean(new_out)
    sigma = statistics.stdev(new_out)
    return mu,sigma

## filter = -1 means we don't filter
def plot_std(mu,sigma,index,out,filter):
    if filter != -1:
        mu, sigma = filter_outlier(out)
    color = []
    color.append("b")
    color.append("y")
    color.append("r")
    question = []
    question.append("What is the weather today")
    question.append("What time is it")
    question.append("Distance from Paris")
    # out.sort()
    # pdf = stats.norm.pdf(out, mu, sigma)
    # plt.plot(out, pdf, color=color[index], label=question[index])
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma),color = color[index], label = question[index])


##ploting graph for Pcap file
def plotSubGraph_2(time,audio,time_s,indicators_x,indicators_y,max_range):
    color = ["b","y","r"]
    question =["What is the weather today","What time is it","Distance from Paris"]
    fig = plt.figure(time)
    plt.ylim(top=max_range)
    plt.ylim(bottom=0)
    plt.title("Web Traffic for set "+str(time+1))
    plt.xlabel("Time")
    plt.ylabel("Total Byte")
    ax = fig.add_subplot(111)
    j = time
    for i in range(1,2):
        byte = audio
        ts = time_s
        start = j * 10 * 3
        end = ((j+1) * 10 * 3)
        start_p = 0;
        for u in range(0,(len(byte))):
            if ts[u] == indicators_x[start]:
                start_p = u
            if ts[u] == indicators_x[end-1]:
                end_p = u
        byte = byte[start_p:end_p + 10]
        ts = ts[start_p:end_p + 10]
        indicators_x = indicators_x[start:end]
        indicators_y = indicators_y[start:end]
        j = 0
        for p in range(0,3):
            subset_x = []
            subset_y = []
            count = 0
            for q in range (0, len(indicators_y)):
                if count == p:
                    subset_y.append(indicators_y[q])
                    subset_x.append(indicators_x[q])
                count +=1
                if count == 3:
                    count = 0
            plt.plot(subset_x,subset_y,'*',color=color[p], label=question[p])
        ax.plot(ts, byte, '.-', color=color[i-1], label=("Set " + str(j+1)))
        plt.legend(loc="best")


##Function used to give prediction using DTW method.
def givePrediction_DTW(q1, q2, q3, val):
    model_1 = myDTW(q1,val)
    model_2 = myDTW(q2, val)
    model_3 = myDTW(q3, val)
    # model_1 = tslearn.metrics.dtw(q1,val)
    # model_2 = tslearn.metrics.dtw(q2, val)
    # model_3 = tslearn.metrics.dtw(q3, val)
    min_num = min(model_1,model_2,model_3)
    if model_1 == min_num:
        return 0
    elif model_2 == min_num:
        return 1
    else:
        return 2

##Function used to gie prediction for Navie Bayer.
## 0 ----> question 1, 1-----> question 2, 2-----> question 3.
def givePredictionNavieBayer(q1_mean, q1_std, q2_mean, q2_std,  q3_mean, q3_std, val, overall_mean, overall_std):
    model_1 = (stats.norm(q1_mean,q1_std).pdf(val))/stats.norm(overall_mean,overall_std).pdf(val)
    model_2 = stats.norm(q2_mean, q2_std).pdf(val)/stats.norm(overall_mean,overall_std).pdf(val)
    model_3 = stats.norm(q3_mean, q3_std).pdf(val)/stats.norm(overall_mean,overall_std).pdf(val)
    max_num = max(model_1,model_2,model_3)
    if model_1 == max_num:
        return 0
    elif model_2 == max_num:
        return 1
    else:
        return 2



##Function used to identify the peak that qualifies as interaction.
## Min bound as the packet size of the peak to be consider as valid peak.
##Lower bound is the seconds look priror to the peak
## Upper bound is the seconds look after the peak.
## Return the valid timestamp of peak, and the total time byte transferred lower bound before the peak + upper bound seconds after the peak
def giveValidPeak(peak_packet_index, min_bound, lower_bound, upper_bound, correct_time,ts,byte,overall_mean, overall_std,blocked=False):
    time_find = []
    byte_find = []
    correctly_time = []
    correctly_byte = []
    c = 0
    valid_time = len(correct_time)
    correct_time_temp = correct_time.copy()
    if blocked == False:
        for i in peak_packet_index:
            # 50000 , 10000 ,40000
            if (byte[i] > min_bound):
                front = False
                end = False
                for w in range(1, upper_bound + 1):
                    if (time_find.__contains__(ts[i] - w)):
                        front = True
                        break
                for w in range(0, lower_bound):
                    if (time_find.__contains__(ts[i] + w)):
                        end = True
                        break
                if (not front and not end):
                    sum = 0
                    for w in range(1, lower_bound + 1):
                        sum += byte[i - w]
                    for w in range(0, upper_bound):
                        sum += byte[i + w]
                    byte_find.append(sum)
                    time_find.append(ts[i])
        for i in range(len(time_find)):
            for j in range(6):
                if correct_time_temp.__contains__(j + time_find[i]) or correct_time_temp.__contains__(time_find[i] - j):
                    correctly_time.append(time_find[i])
                    correctly_byte.append(byte_find[i])
                    if correct_time_temp.__contains__(j + time_find[i]):
                        correct_time_temp.remove(j + time_find[i])
                    elif correct_time_temp.__contains__(time_find[i] - j):
                        correct_time_temp.remove(time_find[i] - j)
                    c += 1
                    break
    else:
        for i in peak_packet_index:
            # 50000
            if (byte[i] > 500):
                byte_find.append(byte[i])
                time_find.append(ts[i])
        for i in range(len(time_find)):
            for j in range(11):
                if correct_time.__contains__(time_find[i] + j):
                    correctly_time.append(time_find[i])
                    correctly_byte.append(byte_find[i])
                    correct_time.remove(j + time_find[i])
                    c += 1
                    break
    print("Identified " + str(len(time_find)) + " There are " + str(
        valid_time) + " total and number of correctly identified is " + str(c))
    return correctly_time,correctly_byte, len(byte_find)


##Same function as the one aboved, but returns valid time stamp of the peak and an array of sequence of byte.
# Each sequence is the byte transfer x - lowerbound, to x + upper bound.
def giveValidPeak_DTW(peak_packet_index, min_bound, lower_bound, upper_bound, correct_time,ts,byte):
    time_find = []
    byte_find = []
    correctly_time = []
    correctly_byte = []
    c = 0
    valid_time = len(correct_time)
    for i in peak_packet_index:
        # 50000 , 10000 ,40000
        if (byte[i] > min_bound):
            front = False
            end = False
            for w in range(1, upper_bound + 1):
                if (time_find.__contains__(ts[i] - w)):
                    front = True
                    break
            for w in range(0, lower_bound):
                if (time_find.__contains__(ts[i] + w)):
                    end = True
                    break
            if (not front and not end):
                sum = []
                for w in range( -1*(lower_bound) -1, upper_bound):
                    sum.append((byte[i + w]))
                byte_find.append(sum)
                time_find.append(ts[i])
    for i in range(len(time_find)):
        for j in range(6):
            if correct_time.__contains__(j + time_find[i]) or correct_time.__contains__(time_find[i] - j):
                correctly_time.append(time_find[i])
                correctly_byte.append(byte_find[i])
                if correct_time.__contains__(j + time_find[i]):
                    correct_time.remove(j + time_find[i])
                elif correct_time.__contains__(time_find[i] - j):
                    correct_time.remove(time_find[i] - j)
                c += 1
                break
    print("Identified " + str(len(time_find)) + " There are " + str(
        valid_time) + " total and number of correctly identified is " + str(c))
    return correctly_time,correctly_byte


##Count the number of interaction that we identified to each of the three question.
def count_correct(q1,q2,q3, myestimate,acceptance_range = 6):
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    for est_time in myestimate:
        for i in range(acceptance_range):
            if (q1.__contains__(est_time + i) or q1.__contains__(est_time - i)):
                correct_1 += 1
                break
    for est_time in myestimate:
        for i in range(acceptance_range):
            if (q2.__contains__(est_time + i) or q2.__contains__(est_time - i)):
                correct_2 += 1
                break
    for est_time in myestimate:
        for i in range(acceptance_range):
            if (q3.__contains__(est_time + i) or q3.__contains__(est_time - i)):
                correct_3 += 1
                break
    return correct_1,correct_2,correct_3

##Function to search for lower bound of line.
def searchForMinBound(peak_packet_index,ts,byte,correct_time,min_Barrier, max_Barrier):
    temp = 10000000000000
    best = -1
    for i in range(min_Barrier,max_Barrier,500):
        correctly_time,correctly_byte,byte_found_length = giveValidPeak(peak_packet_index,i,5,10,correct_time,ts,byte)
        if(len(correctly_byte)< 448):
            continue
        else:
            if byte_found_length < temp:
                temp = byte_found_length
                best = i
    print(str(temp)+ "   " + str(best))


##Function used to make prediction using Navie Bayers. Printing the result.
def TryToIdentify(byte, ts, correct_time, q1_mean, q1_std, q2_mean, q2_std,  q3_mean, q3_std,all_mean,all_std, blocked = False, block_range = 10,):
    correct_q1 = []
    correct_q2 = []
    correct_q3 = []
    q1 = []
    q2 = []
    q3 = []
    real_time = correct_time.copy()
    ## Tries to identify peaks
    peak_packet_index = peakutils.peak.indexes(byte,thres = 0)
    correctly_time,correctly_byte,byte_found_length= giveValidPeak(peak_packet_index,50000,3,8,correct_time,ts,byte,allmean,allstd)
    ## Record predication
    for i in range(len(correctly_time)):
        q_num = givePredictionNavieBayer(q1_mean, q1_std, q2_mean, q2_std,  q3_mean, q3_std, correctly_byte[i],all_mean,all_std)
        if(q_num==0):
            q1.append(correctly_time[i])
        elif(q_num == 1):
            q2.append(correctly_time[i])
        else:
            q3.append(correctly_time[i])
    print("Number to be weather " + str(len(q1)) + ", number to be time " + str(len(q2)) + ", number to be distance " + str(len(q3)))
    count = 0
    for i in range(len(real_time)):
        if(count == 0):
            correct_q1.append(real_time[i])
        elif(count == 1):
            correct_q2.append(real_time[i])
        elif(count == 2):
            correct_q3.append(real_time[i])
        count +=1
        if(count == 3):
            count = 0
    c_1,c_2,c_3= count_correct(correct_q1,correct_q2,correct_q3,q1,acceptance_range= block_range)
    print("Identify as Q1 but belong to q1 "+ str(c_1)+" to q2 "+ str(c_2)+" to q3 "+str(c_3))
    c_1,c_2,c_3 = count_correct(correct_q1, correct_q2, correct_q3, q2, acceptance_range= block_range)
    print("Identify as Q2 but belong to q1 "+ str(c_1)+" to q2 "+ str(c_2)+" to q3 "+str(c_3))
    c_1,c_2,c_3 = count_correct(correct_q1, correct_q2, correct_q3, q3, acceptance_range= block_range)
    print("Identify as Q3 but belong to q1 " + str(c_1) + " to q2 " + str(c_2) + " to q3 " + str(c_3))


## Combined Function that allow you to read both Pcap file and json file.
def readFile(time_file_name, package_file_name,time_stamp_file_name , fileType = "Pcap"):
    if fileType == "json":
        return readJson(package_file_name,time_stamp_file_name);
    time_file = open(time_file_name, "r")
    package_file = open(package_file_name, "r")
    time_stamp_file = open(time_stamp_file_name, "r")
    temptime = []
    temppackage = []
    recorded_time_stamp = []

    for line in time_stamp_file:
        recorded_time_stamp.append(int(float(line.strip())))

    for line in time_file:
        temptime.append(int(float(line.strip())))

    for line in package_file:
        temppackage.append(int(float(line.strip())))
    dic_2 = {}
    ts_2 = []
    init_2 = temptime[0]
    max_num = 0
    for i in range(0, len(temptime)):
        if not dic_2.__contains__(temptime[i] - temptime[0]):
            dic_2[temptime[i] - temptime[0]] = temppackage[i]
        else:
            dic_2[temptime[i] - temptime[0]] += temppackage[i]
        max_num = max(max_num, dic_2[temptime[i] - temptime[0]])

    byte_2 = []

    for i in range(0, temptime[len(temptime) - 1] - temptime[0]):
        ts_2.append(i)
        if dic_2.__contains__(i):
            byte_2.append(dic_2[i])
        else:
            byte_2.append(0)
    indicators_x = []
    indicators_y = []
    for i in range(0, len(recorded_time_stamp)):
        indicators_x.append(recorded_time_stamp[i] - init_2)
        indicators_y.append(max_num * 1.1)
    return ts_2,byte_2,indicators_x,indicators_y,max_num

##Function used to make prediction using DTW. Printing the result.
def IdentifyThroughDTW(byte,ts,indicators_x,time_window_1,time_window_2,time_window_3):
    peak_packet_index = peakutils.peak.indexes(byte, thres=0)
    real_time = indicators_x.copy()
    correctly_time, correctly_byte = giveValidPeak_DTW(peak_packet_index, 50000, 5, 10, indicators_x,ts,byte)
    q1 = []
    q2= []
    q3 = []
    correct_q1 = []
    correct_q2 = []
    correct_q3 = []
    for i in range(len(correctly_time)):
        q_num = givePrediction_DTW(time_window_1, time_window_2, time_window_3, correctly_byte[i])
        if(q_num==0):
            q1.append(correctly_time[i])
        elif(q_num == 1):
            q2.append(correctly_time[i])
        else:
            q3.append(correctly_time[i])
    count = 0
    for i in range(len(real_time)):
        if(count == 0):
            correct_q1.append(real_time[i])
        elif(count == 1):
            correct_q2.append(real_time[i])
        elif(count == 2):
            correct_q3.append(real_time[i])
        count +=1
        if(count == 3):
            count = 0
    # print(q1)
    c_1,c_2,c_3= count_correct(correct_q1,correct_q2,correct_q3,q1)
    print("Identify as Q1 but belong to q1 "+ str(c_1)+" to q2 "+ str(c_2)+" to q3 "+str(c_3))
    c_1,c_2,c_3 = count_correct(correct_q1, correct_q2, correct_q3, q2)
    print("Identify as Q2 but belong to q1 "+ str(c_1)+" to q2 "+ str(c_2)+" to q3 "+str(c_3))
    c_1,c_2,c_3 = count_correct(correct_q1, correct_q2, correct_q3, q3)
    print("Identify as Q3 but belong to q1 " + str(c_1) + " to q2 " + str(c_2) + " to q3 " + str(c_3))

# ts_2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
# byte_2 = [10,5,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,30,5,2,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,0,3,1,0,0,0,0,0,0,0,37,4,2,0,0]
# indicators_x = [0,10,20,30,40,50]
# ts = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]
# byte = [0,0,0,0,0,10,5,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,30,5,2,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,0,3,1,0,0,0,0,0,0,0,37,4,2,0,0,0,0,0,0,0]
#
# ##[0,0,0,0,0,||10,5,0,0,0,|||||0,0,0,0,0||||||,0,3,0,0,0|||||,0,0,0,0,0,||||||30,5,2,0,0,|||||||0,0,0,0,0||||||,0,0,12,0,0|||||||,0,0,0,0,0,0,3,1,0,0,0,0,0,0,0,37,4,2,0,0,0,0,0,0,0]
#
# print(tslearn.metrics.dtw([5,2,6,0,0], [0,30,5,2,0,0]))
# print(tslearn.metrics.dtw([0,30,5,2,0,0],[30,5,2,0,0]))
# indicators_x = [5,15,25,35,45,55]


plt.style.use('seaborn-whitegrid')


##reading the file
ts,byte,indicators_x,indicators_y,max_num= readFile("Audio_Data/SUM/15hours_time_stamp.txt", "Audio_Data/SUM/15hours_package_size.txt","Audio_Data/SUM/ts_3.txt")

##Getting the std, mean, total packet 10 seconds after the initial call for each of the three questions.
a_1,b_1,temp_1,time_window_1 = find_Stats(0,byte,ts,indicators_x, block_range= 10)
a_f_1, b_f_1 = filter_outlier(temp_1)
a_2,b_2,temp_2,time_window_2 = find_Stats(1,byte,ts,indicators_x,block_range= 10)
a_f_2, b_f_2 = filter_outlier(temp_2)
a_3,b_3,temp_3,time_window_3 = find_Stats(2,byte,ts,indicators_x,block_range= 10)
a_f_3, b_f_3 = filter_outlier(temp_3)
allstd, allmean,temp_4,time_window_4 = find_Stats(0,byte,ts,indicators_x,all= True)


##Function used to group them into bins.
# byte_block =[]
# time_block = []
# block_range = 10
# for i in range(0, len(ts),block_range) :
#     sum = 0
#     j = 0
#     while (j < block_range and i+j < len(ts)):
#         sum += byte[i+j]
#         j+=1
#     time_block.append(i)
#     byte_block.append(sum)

# print(byte_block)
# print(time_block)

## in bins ###
# a_1,b_1,temp_1 = find_Stats(0,byte_block,time_block,indicators_x,in_block= True,block_range = block_range)
# a_f_1, b_f_1 = filter_outlier(temp_1)
# a_2,b_2,temp_2 = find_Stats(1,byte_block,time_block,indicators_x,in_block= True, block_range = block_range)
# a_f_2, b_f_2 = filter_outlier(temp_2)
# a_3,b_3,temp_3 = find_Stats(2,byte_block,time_block,indicators_x,in_block= True,block_range = block_range)
# a_f_3, b_f_3 = filter_outlier(temp_3)
# find_Stats(0,byte_block,time_block,indicators_x,in_block= True)
# find_Stats(1,byte_block,time_block,indicators_x,in_block= True)
# find_Stats(2,byte_block,time_block,indicators_x,in_block= True)




##ANOVA
print(stats.f_oneway(temp_1,temp_2,temp_3))


##
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

data = [[0 for x in range(len(time_window_1))] for y in range(3)]
for i in range (3):
    for j in range (len(time_window_1)):
        if(i == 0):
            data[i][j] = time_window_1[j]
        if (i == 1):
            data[i][j] = time_window_2[j]
        if (i ==2):
            data[i][j] = time_window_3[j]



# col = [x for x in range(len(time_window_1))]
# row = ["Q1","Q2","Q3"]
# fig, ax = plt.subplots()
# im, cbar = heastmap(np.array(data), row, col, ax=ax,
#                    cmap="YlOrBr", cbarlabel="Average Byte being send 10 seconds after the audio has being played")
#
# # ax.set_xticks(np.arange(len(columns)))
# # ax.set_yticks(np.arange(len(rows)))
# # ax.set_xticklabels(columns)
# # ax.set_yticklabels(rows)
# # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
# #          rotation_mode="anchor")
# # cax = ax.matshow(data, interpolation='nearest')
# # fig.colorbar(cax)
# fig.tight_layout()
# plt.show()

###################################
##without Outlier
#
# ts,byte,indicators_x,indicators_y,max_num= readFile("Audio_Data/SUM/1hour_mix_time_stamp.txt", "Audio_Data/SUM/1hour_mix_packcket.txt","Audio_Data/SUM/ts_6.txt")
# #
TryToIdentify(byte, ts, indicators_x, b_f_1, a_f_1, b_f_2, a_f_2, b_f_3, a_f_3,allmean,allstd)
# IdentifyThroughDTW(byte,ts,indicators_x,time_window_1,time_window_2,time_window_3)
##in block
# TryToIdentify(byte_block, time_block, indicators_x, b_f_1, a_f_1, b_f_2, a_f_2, b_f_3, a_f_3, blocked = True, block_range = block_range+1)
##with
# TryToIdentify(byte_2, ts_2, indicators_x, b_1, a_1, b_2, a_2, b_3, a_3)


############normal disturtbiotn ###########################3
# for i in range(0,3):
#     std, mean,temp = find_Stats(i,byte_block,time_block,indicators_x)
#     plot_std(mean,std,i,temp,1)
# plt.ylim(top = 0.00005)
# plt.ylim(bottom = 0)
# plt.xlim(right=500000)
# plt.xlim(left = 0)
# plt.legend(loc = "best")
# plt.title("Total Normal Disturbtion of Total Data With Outlier")
#######################################################################

########## PLOT everything #############
# for i in range(0,1):
#     plotSubGraph_2(i,byte,ts,indicators_x,indicators_y,max_num*1.5)


#####################################

### BOX PLOT #######
#
# for i in range(0,3) :
#     question = []
#     question.append("What is the weather today")
#     question.append("What time is it")
#     question.append("Distance from Paris")
#     std,mean,temp = find_Medium(i,byte_2,ts_2,indicators_x)
#     fig = plt.figure(i)
#     plt.title("Box plot for question \"" +question[i] +"\"")
#     plt.boxplot(temp)
#     plt.ylabel("Send byte starting from when question was asked")
#     plt.ylim(top=810000)
#     plt.ylim(bottom=0)
############

plt.show()

#### Tries to identify the best min_bound ##########
# min_when_connection =10000000000000000
# max_when_no = -1
# p = 0
# for i in range(len(ts_2)):
#     if p < len(indicators_x) and within10(ts_2[i], indicators_x):
#         if(byte_2[i]!= 0):
#             min_when_connection = min (min_when_connection, byte_2[i])
#     else:
#         max_when_no = max(max_when_no,byte_2[i])
#
# print(min_when_connection);
# print(max_when_no);
# min_distance = 2123123213213123
# for i in range(79501,85000,500):
#     output = TryToIdentify(byte_2, ts_2, indicators_x, b_1, a_1, b_2, a_2, b_3, a_3, i)
#     if(output<min_distance):
#         min_distance = output
#         min_val = i
# print(min_val)
# print(min_distance)
####################################################