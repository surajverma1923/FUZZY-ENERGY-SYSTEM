import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as pp
import datetime
#load dataset
path = "energydata_complete.csv"
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False)

for i in range(9):
    i=i+1
temp_1 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_1')
temp_1['verycold'] = fuzz.zmf(temp_1.universe, -7, 3)
temp_1['cold'] = fuzz.trimf(temp_1.universe,[0,5,10])
temp_1['cool'] = fuzz.trimf(temp_1.universe,[8,12,15])
temp_1['warm'] = fuzz.trimf(temp_1.universe,[12,17,20])
temp_1['hot'] = fuzz.smf(temp_1.universe, 18, 30)


temp_2 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_2')
temp_2['verycold'] = fuzz.zmf(temp_2.universe, -7, 3)
temp_2['cold'] = fuzz.trimf(temp_2.universe,[0,5,10])
temp_2['cool'] = fuzz.trimf(temp_2.universe,[8,12,15])
temp_2['warm'] = fuzz.trimf(temp_2.universe,[12,17,20])
temp_2['hot'] = fuzz.smf(temp_2.universe, 18, 30)

temp_3 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_3')
temp_3['verycold'] = fuzz.zmf(temp_3.universe, -7, 3)
temp_3['cold'] = fuzz.trimf(temp_3.universe,[0,5,10])
temp_3['cool'] = fuzz.trimf(temp_3.universe,[8,12,15])
temp_3['warm'] = fuzz.trimf(temp_3.universe,[12,17,20])
temp_3['hot'] = fuzz.smf(temp_3.universe, 18, 30)

temp_4 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_4')
temp_4['verycold'] = fuzz.zmf(temp_4.universe, -7, 3)
temp_4['cold'] = fuzz.trimf(temp_4.universe,[0,5,10])
temp_4['cool'] = fuzz.trimf(temp_4.universe,[8,12,15])
temp_4['warm'] = fuzz.trimf(temp_4.universe,[12,17,20])
temp_4['hot'] = fuzz.smf(temp_4.universe, 18, 30)

temp_5 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_5')
temp_5['verycold'] = fuzz.zmf(temp_5.universe, -7, 3)
temp_5['cold'] = fuzz.trimf(temp_5.universe,[0,5,10])
temp_5['cool'] = fuzz.trimf(temp_5.universe,[8,12,15])
temp_5['warm'] = fuzz.trimf(temp_5.universe,[12,17,20])
temp_5['hot'] = fuzz.smf(temp_5.universe, 18, 30)

temp_6 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_6')
temp_6['verycold'] = fuzz.zmf(temp_6.universe, -7, 3)
temp_6['cold'] = fuzz.trimf(temp_6.universe,[0,5,10])
temp_6['cool'] = fuzz.trimf(temp_6.universe,[8,12,15])
temp_6['warm'] = fuzz.trimf(temp_6.universe,[12,17,20])
temp_6['hot'] = fuzz.smf(temp_6.universe, 18, 30)

temp_7 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_7')
temp_7['verycold'] = fuzz.zmf(temp_7.universe, -7, 3)
temp_7['cold'] = fuzz.trimf(temp_7.universe,[0,5,10])
temp_7['cool'] = fuzz.trimf(temp_7.universe,[8,12,15])
temp_7['warm'] = fuzz.trimf(temp_7.universe,[12,17,20])
temp_7['hot'] = fuzz.smf(temp_7.universe, 18, 30)

temp_8 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_8')
temp_8['verycold'] = fuzz.zmf(temp_8.universe, -7, 3)
temp_8['cold'] = fuzz.trimf(temp_8.universe,[0,5,10])
temp_8['cool'] = fuzz.trimf(temp_8.universe,[8,12,15])
temp_8['warm'] = fuzz.trimf(temp_8.universe,[12,17,20])
temp_8['hot'] = fuzz.smf(temp_8.universe, 18, 30)

temp_9 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_9')
temp_9['verycold'] = fuzz.zmf(temp_9.universe, -7, 3)
temp_9['cold'] = fuzz.trimf(temp_9.universe,[0,5,10])
temp_9['cool'] = fuzz.trimf(temp_9.universe,[8,12,15])
temp_9['warm'] = fuzz.trimf(temp_9.universe,[12,17,20])
temp_9['hot'] = fuzz.smf(temp_9.universe, 18, 30)

humidity_1 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_1')
humidity_1['dry'] = fuzz.trapmf(humidity_1.universe,[0,0,25,40])
humidity_1['comfortable'] = fuzz.trapmf(humidity_1.universe,[25,40,60,75])
humidity_1['humid'] = fuzz.trapmf(humidity_1.universe,[60,75,100,100])


humidity_2 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_2')
humidity_2['dry'] = fuzz.trapmf(humidity_2.universe,[0,0,25,40])
humidity_2['comfortable'] = fuzz.trapmf(humidity_2.universe,[25,40,60,75])
humidity_2['humid'] = fuzz.trapmf(humidity_2.universe,[60,75,100,100])

humidity_3 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_3')
humidity_3['dry'] = fuzz.trapmf(humidity_3.universe,[0,0,25,40])
humidity_3['comfortable'] = fuzz.trapmf(humidity_3.universe,[25,40,60,75])
humidity_3['humid'] = fuzz.trapmf(humidity_3.universe,[60,75,100,100])

humidity_4 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_4')
humidity_4['dry'] = fuzz.trapmf(humidity_4.universe,[0,0,25,40])
humidity_4['comfortable'] = fuzz.trapmf(humidity_4.universe,[25,40,60,75])
humidity_4['humid'] = fuzz.trapmf(humidity_4.universe,[60,75,100,100])

humidity_5 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_5')
humidity_5['dry'] = fuzz.trapmf(humidity_5.universe,[0,0,25,40])
humidity_5['comfortable'] = fuzz.trapmf(humidity_5.universe,[25,40,60,75])
humidity_5['humid'] = fuzz.trapmf(humidity_5.universe,[60,75,100,100])

humidity_6 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_6')
humidity_6['dry'] = fuzz.trapmf(humidity_6.universe,[0,0,25,40])
humidity_6['comfortable'] = fuzz.trapmf(humidity_6.universe,[25,40,60,75])
humidity_6['humid'] = fuzz.trapmf(humidity_6.universe,[60,75,100,100])

humidity_7 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_7')
humidity_7['dry'] = fuzz.trapmf(humidity_7.universe,[0,0,25,40])
humidity_7['comfortable'] = fuzz.trapmf(humidity_7.universe,[25,40,60,75])
humidity_7['humid'] = fuzz.trapmf(humidity_7.universe,[60,75,100,100])

humidity_8 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_8')
humidity_8['dry'] = fuzz.trapmf(humidity_8.universe,[0,0,25,40])
humidity_8['comfortable'] = fuzz.trapmf(humidity_8.universe,[25,40,60,75])
humidity_8['humid'] = fuzz.trapmf(humidity_8.universe,[60,75,100,100])

humidity_9 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_9')
humidity_9['dry'] = fuzz.trapmf(humidity_9.universe,[0,0,25,40])
humidity_9['comfortable'] = fuzz.trapmf(humidity_9.universe,[25,40,60,75])
humidity_9['humid'] = fuzz.trapmf(humidity_9.universe,[60,75,100,100])

windspeed = ctrl.Antecedent(np.arange(0,21,0.01), 'windspeed')

windspeed['Low'] = fuzz.trimf(windspeed.universe,[0,0,4])
windspeed['Medium'] = fuzz.trimf(windspeed.universe,[3,5,7])
windspeed['High'] = fuzz.trapmf(windspeed.universe,[6,10,20,20])

visibility = ctrl.Antecedent(np.arange(0,76,0.01), 'visibility')

visibility['Low'] = fuzz.trimf(visibility.universe,[0,0,40])
visibility['Medium'] = fuzz.trimf(visibility.universe,[30,45,60])
visibility['High'] = fuzz.trapmf(visibility.universe,[50,65,75,75])

pressure = ctrl.Antecedent(np.arange(700,801,0.01), 'pressure')

pressure['Low'] = fuzz.trimf(pressure.universe,[700,700,740])
pressure['Medium'] = fuzz.trimf(pressure.universe,[720,750,780])
pressure['High'] = fuzz.trapmf(pressure.universe,[760,790,800,800])

consumption = ctrl.Consequent(np.arange(0,1201,1), 'consumption')

consumption['Low'] = fuzz.trimf(consumption.universe,[0,0,300])
consumption['Medium'] = fuzz.trimf(consumption.universe,[100,300,500])
consumption['High'] = fuzz.trimf(consumption.universe,[300,500,700])

consumption['VeryHigh'] = fuzz.trapmf(consumption.universe,[500,800,1200,1200])

consumption.view()
pp.savefig(r'consumption_fuzz.png',dpi=300,bbox_inches="tight")



#READ AND PARSE RULES INTO THE SYSTEM
rulefile=open("paths.txt",'r')
rulelines=rulefile.readlines()

#PARSE RULES INTO CONSEQUENTS AND ANTECEDENTS/// CREATING RULEBASE 
count=0
consequent=""
antecedent=""
rule_consequent=""
rulebase=[]
ctrlargs=""
ctrlsys=""
for line in rulelines:
    ruleunparsed=line.strip()
    rulecomponents=ruleunparsed.split("&")
    for r in rulecomponents:
        if r!="":
            component_split=r.split(":")
            if component_split[1]!="":
                antecedent_parts=component_split[0].split("-")
                if component_split[1]=="  False":
                    antecedent=antecedent +"~"+ antecedent_parts[0] + "['"+antecedent_parts[1]+"']" + "&"
                else:
                    antecedent=antecedent + antecedent_parts[0] + "['"+antecedent_parts[1]+"']" + "&"
              
            else:
                consequent="consumption['" + component_split[0] + "']"
                
    antecedent=antecedent[:-1]
    rule="rule"+str(count)+"="+"ctrl.Rule("+antecedent+","+consequent+")"
    ctrlargs=ctrlargs+"rule"+str(count)+","
  
    rulebase.append(rule)
    count=count+1
    consequent=""
    antecedent=""
    #

ctrlargs=ctrlargs[:-1]
ctrlsys="consumption_ctrl = ctrl.ControlSystem(["+ctrlargs+"])"
print(ctrlsys)

#RULE EXECUTION
for ru in rulebase:
    
    exec(ru)

exec(ctrlsys)
exec("consumption_simulation = ctrl.ControlSystemSimulation(consumption_ctrl)")


for j in range(dataset['T1'].shape[0]):
    if any("temp_1" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_1']="+str(dataset['T1'].values[j]))
    else:
        print("temp_1 Not in rulebase")
    
    if any("temp_2" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_2']="+str(dataset['T2'].values[j]))
    else:
        print("temp_2 Not in rulebase")
    
    if any("temp_3" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_3']="+str(dataset['T3'].values[j]))
    else:
        print("temp_3 Not in rulebase")
        
    if any("temp_4" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_4']="+str(dataset['T4'].values[j]))
    else:
        print("temp_4 Not in rulebase")
        
    if any("temp_5" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_5']="+str(dataset['T5'].values[j]))
    else:
        print("temp_5 Not in rulebase")
        
    if any("temp_6" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_6']="+str(dataset['T6'].values[j]))
    else:
        print("temp_6 Not in rulebase")
    if any("temp_7" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_7']="+str(dataset['T7'].values[j]))
    else:
        print("temp_7 Not in rulebase")
    if any("temp_8" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_8']="+str(dataset['T8'].values[j]))
    else:
        print("temp_8 Not in rulebase")
    if any("temp_9" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_9']="+str(dataset['T9'].values[j]))
    else:
        print("temp_9 Not in rulebase")
    
    #Bind Humidity/////////////////////////////////////////////////////
    if any("humidity_1" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_1']="+str(dataset['RH_1'].values[j]))
    else:
        print("humidity_1 Not in rulebase")
    
    if any("humidity_2" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_2']="+str(dataset['RH_2'].values[j]))
    else:
        print("humidity_2 Not in rulebase")
    
    if any("humidity_3" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_3']="+str(dataset['RH_3'].values[j]))
    else:
        print("humidity_3 Not in rulebase")
        
    if any("humidity_4" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_4']="+str(dataset['RH_4'].values[j]))
    else:
        print("humidity_4 Not in rulebase")
        
    if any("humidity_5" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_5']="+str(dataset['RH_5'].values[j]))
    else:
        print("humidity_5 Not in rulebase")
        
    if any("humidity_6" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_6']="+str(dataset['RH_6'].values[j]))
    else:
        print("humidity_6 Not in rulebase")
    if any("humidity_7" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_7']="+str(dataset['RH_7'].values[j]))
    else:
        print("humidity_7 Not in rulebase")
    if any("humidity_8" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_8']="+str(dataset['RH_8'].values[j]))
    else:
        print("humidity_8 Not in rulebase")
    if any("humidity_9" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_9']="+str(dataset['RH_9'].values[j]))
    else:
        print("humidity_9 Not in rulebase")
    
    
    #Bind windspeed, visibility, pressure
    if any("windspeed" in inputterm for inputterm in rulebase):    
        exec("consumption_simulation.input['windspeed']="+str(dataset['Windspeed'].values[j]))
    else:
        print("windspeed Not in rulebase")
    if any("visibility" in inputterm for inputterm in rulebase):    
        exec("consumption_simulation.input['visibility']="+str(dataset['Visibility'].values[j]))
    else:
        print("visibility Not in rulebase")
    if any("pressure" in inputterm for inputterm in rulebase):    
        exec("consumption_simulation.input['pressure']="+str(dataset['Press_mm_hg'].values[j]))
    else:
        print("pressure Not in rulebase")
    
  

    #COMPUTATION
    begin_time=datetime.datetime.now()
    exec("consumption_simulation.compute()")
    print(datetime.datetime.now()-begin_time)
    
    exec("print(consumption_simulation.output['consumption'])")
    exec("consumption.view(sim=consumption_simulation)")
    pp.savefig(r'consumption_simulation.png',dpi=300,bbox_inches="tight")
    break

