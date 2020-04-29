
from flask import Flask
from flask import request
from flask import render_template

import numpy as np
from scipy.stats import poisson

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd





app = Flask(__name__)

app.config["DEBUG"]  = True

@app.route('/')
def my_form():
    return render_template("form.html")

@app.route('/', methods=['POST'])
def my_form_post():
    if 'predict' in request.form:
        nv = int(request.form['num1'])
        na = int(request.form['num2'])
        print(nv, na)
        S, S_error = prediction(nv, na)
        return "<p> Expected number of new ventilators: " + str(S) + "</p><p> 99% confidence interval: "+ str(S_error) +"</p>"
    else:
        lstring = request.form['lambda']
        lambda_t = [ int(item) for item in lstring.split(',')]
        runSim(lambda_t)
        return render_template("image.html", image="/static/images/sim.png")

def poisson_interval(lam, prob=0.99):
    """
    Compute confidence interval for Y ~ Poisson(lam): [0, k] by Chernoff's bound
    :param lam: Poisson distr. mean
    :param prob: P(Y in [0, k])< prob
    :return: k
    """
    k = int(poisson.ppf(prob, lam))
    return k

def prediction(non_vent_patients, new_admissions, q2=1/7, q3=1/5, p1=0.75, p2=0.05, p3=0.2, los1=11, los2=13, los3=5):

    p1_tilde = p1*los1/(p1*los1+p2*los2+p3*los3)
    p2_tilde = p2*los2/(p1*los1+p2*los2+p3*los3)
    p3_tilde = p3*los3/(p1*los1+p2*los2+p3*los3)

    S_mean =  (q2*p2_tilde+q3*p3_tilde)*non_vent_patients + (q2*p2+q3*p3)*new_admissions
    S = int(S_mean)
    S_error = poisson_interval(S_mean)
    return S, S_error

class PatientMix(object):
    """docstring for PatientMix"""

    def __init__(self, lambda_t, p1_mixture=0.5, p1_LOS_1 = 3, p1_LOS_2 = 10, prevent_LOS=5, p2_vent_LOS=11, p3_vent_LOS=6, initialize_p1=0,
                 initialize_p2=0, initialize_p3=0, initialize_v=0, initialize_v2=0, initialzie_v3=0, days=None):
        if days == None:
            self.days = len(lambda_t) - 1
        else:
            self.days = days
        self.lambda_t = lambda_t
        self.p1_mixture = p1_mixture
        self.p1_LOS_1 = p1_LOS_1
        self.p1_LOS_2 = p1_LOS_2
        self.p1_LOS = self.p1_mixture * self.p1_LOS_1 + (1 - self.p1_mixture) * self.p1_LOS_2
        self.prevent_LOS = prevent_LOS
        self.p2_vent_LOS = p2_vent_LOS
        self.p3_vent_LOS = p3_vent_LOS
        self.patient1 = [0] * (self.days + 1)
        self.patient2 = [0] * (self.days + 1)
        self.patient3 = [0] * (self.days + 1)
        self.ventilator = [0] * (self.days + 1)
        self.ventilator2 = [0] * (self.days + 1)
        self.ventilator3 = [0] * (self.days + 1)
        self.non_vent = [0] * (self.days + 1)
        self.new_vent = [0] * (self.days + 1)
        self.new_vent_from_hosp = [0] * (self.days + 1)
        self.new_vent_admit = [0] * (self.days + 1)
        self.released_vent = [0] * (self.days + 1)

        self.p1 = 0.7
        self.p2 = 0.06
        self.p3 = 0.24

        self.initialize(initialize_p1, initialize_p2, initialize_p3, initialize_v, initialize_v2, initialzie_v3)

        for i in range(1, self.days + 1):
            self.simulate_day(i)
        # print(self.patient1, self.patient2, self.patient3)

        # p_hat and p_hat_little are measuring the proportion of people not on vents
        self.total_patients_end = self.patient1[-1] + self.patient2[-1] + self.patient3[-1]
        self.p_hat = [self.patient1[-1] / self.total_patients_end, self.patient2[-1] / self.total_patients_end,
                      self.patient3[-1] / self.total_patients_end]

        self.p_hat_little = [
            self.p1 * self.p1_LOS / (self.p1 * self.p1_LOS + self.p2 * self.prevent_LOS + self.p3 * self.prevent_LOS),
            self.p2 * self.prevent_LOS / (
                        self.p1 * self.p1_LOS + self.p2 * self.prevent_LOS + self.p3 * self.prevent_LOS),
            self.p3 * self.prevent_LOS / (
                        self.p1 * self.p1_LOS + self.p2 * self.prevent_LOS + self.p3 * self.prevent_LOS)]

        for i in range(1, self.days + 1):
            self.non_vent[i] = self.patient1[i] + self.patient2[i] + self.patient3[i]

    def new_patient1(self, start_date, LOS):
        for date in range(start_date, min(start_date + LOS, self.days + 1)):
            self.patient1[date] += 1

    def new_patient2(self, start_date, LOS_hosp, LOS_vent):
        for date in range(start_date, min(start_date + LOS_hosp, self.days + 1)):
            self.patient2[date] += 1
        for date in range(min(start_date + LOS_hosp, self.days + 1),
                          min(start_date + LOS_hosp + LOS_vent, self.days + 1)):
            self.ventilator[date] += 1
            self.ventilator2[date] += 1
        if start_date + LOS_hosp <= self.days:
            self.new_vent[start_date+LOS_hosp] += 1
            if LOS_hosp == 0:
                self.new_vent[start_date+LOS_hosp] += 1
            else:
                self.new_vent_from_hosp[start_date + LOS_hosp] += 1

        if start_date + LOS_hosp + LOS_vent <= self.days:
            self.released_vent[start_date + LOS_hosp + LOS_vent] += 1

    def new_patient3(self, start_date, LOS_hosp, LOS_vent):
        for date in range(start_date, min(start_date + LOS_hosp, self.days + 1)):
            self.patient3[date] += 1
        for date in range(min(start_date + LOS_hosp, self.days + 1),
                          min(start_date + LOS_hosp + LOS_vent, self.days + 1)):
            self.ventilator[date] += 1
            self.ventilator3[date] += 1

        if start_date + LOS_hosp <= self.days:
            self.new_vent[start_date+LOS_hosp] += 1
            if LOS_hosp == 0:
                self.new_vent[start_date+LOS_hosp] += 1
            else:
                self.new_vent_from_hosp[start_date + LOS_hosp] += 1


        if start_date + LOS_hosp + LOS_vent <= self.days:
            self.released_vent[start_date + LOS_hosp + LOS_vent] += 1

    def initialize(self, patient1=0, patient2=0, patient3=0, ventilator=0, ventilator2=0, ventilator3=0):
        self.patient1[0] = patient1
        self.patient2[0] = patient2
        self.patient3[0] = patient3
        self.ventilator[0] = ventilator
        self.ventilator2[0] = ventilator2
        self.ventilator3[0] = ventilator3

    def simulate_day(self, date):
        num_patients = self.lambda_t[date]
        # num_patients = np.random.poisson(self.lambda_t[date])  # random poisson arrivals
        new_patients = np.random.multinomial(num_patients, [self.p1, self.p2, self.p3], size=1)
        # print(new_patients, new_patients[0])
        [new_admit1, new_admit2, new_admit3] = new_patients[0]
        for _ in range(new_admit1):
            self.new_patient1(date, p1_LOS(self.p1_mixture, self.p1_LOS_1, self.p1_LOS_2))
        for _ in range(new_admit2):
            self.new_patient2(date, patient_LOS(self.prevent_LOS), patient_LOS(self.p2_vent_LOS))
        for _ in range(new_admit3):
            self.new_patient3(date, patient_LOS(self.prevent_LOS), patient_LOS(self.p3_vent_LOS))


def patient_LOS(avg):
    # return np.random.poisson(avg)
    return np.random.geometric(1 / (avg))

def p1_LOS(prob_1, LOS_1, LOS_2):
    if np.random.uniform() < prob_1:
        return np.random.geometric(1 / LOS_1)
    else:
        return np.random.geometric(1 / LOS_2)



# def Simulation(sim_count, lambda_t, p1_mixture=0.71, p1_LOS_1=1.9, p1_LOS_2=29.8, prevent_LOS=4.3, p2_vent_LOS=12, p3_vent_LOS=8):
def Simulation(sim_count, lambda_t, p1_mixture=0.57, p1_LOS_1=1.1, p1_LOS_2=21.8, prevent_LOS=5.1, p2_vent_LOS=12.3, p3_vent_LOS=6.4):
    matrix_results = list()
    matrix_results2 = list()
    matrix_results3 = list()
    matrix_results4 = list()
    for _ in range(sim_count):
        obj = PatientMix(lambda_t, p1_mixture=p1_mixture, p1_LOS_1 = p1_LOS_1, p1_LOS_2 = p1_LOS_2, prevent_LOS=prevent_LOS, p2_vent_LOS=p2_vent_LOS, p3_vent_LOS=p3_vent_LOS)
        matrix_results.append(obj.ventilator)
        matrix_results2.append(obj.non_vent)
        matrix_results3.append(obj.new_vent)
        matrix_results4.append(obj.released_vent)
    patients_on_vent = matrix_results
    patients_not_on_vent = matrix_results2
    new_vents = matrix_results3
    released_vents = matrix_results4
    # patients_on_vent = np.mean(np.array(matrix_results), axis=0)
    # patients_not_on_vent = np.mean(np.array(matrix_results2), axis=0)
    # new_vents = np.mean(np.array(matrix_results3), axis=0)
    # released_vents = np.mean(np.array(matrix_results4), axis=0)
    # print(patients_on_vent)
    # print(new_vents)
    # print(released_vents)
    # print(patients_not_on_vent)
    return patients_not_on_vent, patients_on_vent, new_vents, released_vents

def runSim(lambda_t):
    #  lambda_t = [0, 12, 7, 10, 15, 11, 9, 15, 36, 47, 71, 74, 138, 154, 188, 302, 341, 434, 512, 608, 641, 680, 981, 1056, 1219, 1310, 1267, 1222, 1302, 1550, 1432, 1431, 1619, 1618, 1355, 1340, 1606, 1360, 1279, 991, 890, 616, 357, 152, 29]
    # Run Simulation
    patients_not_on_vent, patients_on_vent, new_vents, released_vents = Simulation(100, lambda_t)

    # Processing Simulation Results
    dates = list(range(len(lambda_t)))
    dates_graph = list()
    on_vent = list()
    not_on_vent = list()
    for index in range(len(patients_on_vent)):
        on_vent = on_vent + patients_on_vent[index]
        not_on_vent = not_on_vent + patients_not_on_vent[index]
        dates_graph = dates_graph + dates
    total_patients = list()
    for index in range(len(on_vent)):
        total_patients.append(on_vent[index] + not_on_vent[index])

    # Graphing Results
    boxplot_colors = ['Simulation']*len(on_vent)
    sns.boxplot(x=dates_graph, y=on_vent, hue=boxplot_colors)
    plt.legend()
    plt.xlabel('Date', fontsize='x-large')
    plt.ylabel('Ventilators', fontsize='x-large')
    plt.title('Simulated Ventilator Demand', fontsize='xx-large')

    # plt.show()
    plt.savefig("/home/davidpwilliamson/mysite/static/images/sim.png")
