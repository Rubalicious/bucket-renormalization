from tools import *
from mean_field import MeanField
from belief_propagation import BeliefPropagation
import copy
import json

def test0():
    nb_vars = 70
    delta = 1.0

    for i in range(2,20):
        print(10*i)
        toy = generate_complete(10*i, delta)
        # toy = generate_star(10*i)
        t1 = time.time()
        Z = BucketRenormalization(toy, ibound=10).run(max_iter=1)
        t2 = time.time()

        print(Z)
        print(t2-t1)
    quit()

def testing_run_time():
    init_inf = [0]
    runtime = []
    TAUS = [60, 70, 80, 90, 100, 110, 120]
    for H_a in [1.0]:
        for MU in [0.00138985]:
            for TAU in TAUS:
                G = extract_seattle_data(TAU, MU)
                seattle = generate_seattle(G, init_inf, H_a)
                # =====================================
                # Compute partition function for Seattle GM
                # =====================================
                t1 = time.time()
                Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
                t2 = time.time()
                # =====================================
                print('partition function = {}'.format(Z))
                print('time taken for GBR = {}'.format(t2-t1))
                runtime.append(t2-t1)
    plt.plot(TAUS, runtime)
    plt.ylabel("GBR Runtime")
    plt.xlabel(R"minium threshold $\tau$")
    plt.show()
    quit()

def testing_partition_function_dependence_on_TAU():
    init_inf = [0]
    pf = []
    TAUS = [70, 80, 90, 100, 110, 120]
    for H_a in [1.0]:
        for MU in [0.00138985]:
            for TAU in TAUS:
                G = extract_seattle_data(TAU, MU)
                seattle = generate_seattle(G, init_inf, H_a)
                # =====================================
                # Compute partition function for Seattle GM
                # =====================================
                t1 = time.time()
                Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
                t2 = time.time()
                # =====================================
                print('partition function = {}'.format(Z))
                print('time taken for GBR = {}'.format(t2-t1))
                pf.append(Z)
    plt.plot(TAUS, pf)
    plt.ylabel("Z Value")
    plt.xlabel(R"minium threshold $\tau$")
    pls.savefig("Z_dependence_on_tau.png")
    plt.show()
    quit()

def recreate_pandemy(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.0005, ibound=10):
    # extract data or a subset of the data
    J = extract_data_top20(case, MU)


    # create a complete graph GM
    model = generate_graphical_model(case, J, init_inf, H_a, condition_on_init_inf=False)

    # condition on the initial seed 'V0'
    init_inf = [ith_object_name('V',var) for var in init_inf]
    conditioned_on_init = condition_on_seeds_from(model, init_inf, in_place=False)

    logZ = BucketElimination(conditioned_on_init).run()
    print("logZ computed = {}".format(logZ))

    remaining = [var for var in model.variables if var not in init_inf]
    # print(remaining)
    ratios = []
    filename = 'ratios.csv'
    for var in remaining:
        seeds = copy.copy(init_inf)
        seeds.append(var)

        conditioned_on_init_and_var = condition_on_seeds_from(model, seeds, in_place=False)
        logZi = BucketElimination(conditioned_on_init_and_var).run()

        H_val = conditioned_on_init.get_factor(var.replace('V','B')).log_values[0]
        marg_prob = np.exp(H_val)*np.exp(logZi)/np.exp(logZ)
        print(marg_prob)
        utils.append_to_csv(filename, [var, marg_prob])


def plot_PM_vs_ratios():
    pm_data = utils.read_csv('PM.csv')
    PM = [p[1] for p in pm_data][1:]
    ratio_data = utils.read_csv('ratios.csv')
    ratios = [float(p[1]) for p in ratio_data]
    print("sum = {}".format(np.sum(ratios)))
    ratio1_data = utils.read_csv('ratios1.csv')
    ratios1 = [p[1] for p in ratio1_data]
    plt.plot(range(len(PM)), PM,'*-', label='logZi/logZ')
    plt.plot(range(len(ratios)), ratios,'--', label='e^{-h_i}Zi/Z')
    # plt.plot(range(len(ratios1)), ratios1,'o-', label='recreated')
    plt.legend()
    plt.show()

def marg_prob_formula(H_a, MU):
    # extract data or a subset of the data
    J = extract_data_top20(case, MU)

    # create a complete graph GM
    model = generate_graphical_model(case, J, H_a)

def implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.002, ibound=10):
    '''
        This runs BP, MF, GBR, and BE
    '''
    # extract data or a subset of the data
    # J = extract_data_top10(MU)
    # J = extract_data_top20(case, MU=MU)
    J = extract_data(case, MU)
    print(np.shape(J))

    # create a complete graph GM
    model = generate_graphical_model(case, J, H_a)

    # choose algorithm
    if alg == 'BP':
        # condition_on_seeds_from(model, init_inf)
        init_inf = [ith_object_name('V',var) for var in init_inf]
        conditioned_on_init = condition_on_seeds_from(model, init_inf)
        print("before")
        logZ = BeliefPropagation(conditioned_on_init).run(max_iter=100, converge_thr=1e-4)
        print("after")
        CALIs = []
        for index in range(len(J)):
            if ith_object_name('V',index) not in init_inf:
                CALI = 2*logZ['marginals']['MARGINAL_V{}'.format(index)]-1
                CALIs.append(CALI[1])
            else:
                CALIs.append(+1)
        return CALIs
        filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg, case, init_inf, H_a, MU)
    elif alg == 'MF':
        # condition_on_seeds_from(model, init_inf)
        init_inf = [ith_object_name('V',var) for var in init_inf]
        conditioned_on_init = condition_on_seeds_from(model, init_inf)
        logZ = MeanField(conditioned_on_init).run()
        for index in range(len(J)):
            if ith_object_name('V',index) not in init_inf:
                CALI = 2*logZ['marginals']['MARGINAL_V{}'.format(index)]-1
                CALIs.append(CALI[1])
            else:
                CALIs.append(+1)
        return CALIs
    elif alg == 'GBR':
        # approximate
        CALIs = compute_marginals( model, (init_inf, H_a, MU, ibound), alg = alg)

        # filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg+str(ibound), case, init_inf, H_a, MU)
        # utils.append_to_csv(filename, ['CALIs','GM logZ'])
        # utils.append_to_csv(filename, [CALIs, logZ])
        return CALIs
    elif alg == 'BE':
        # exact
        CALIs = compute_marginals( model, (init_inf, H_a, MU, ibound), alg = alg)
        # filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg, case, init_inf, H_a, MU)
        # utils.append_to_csv(filename, ['CALIs','GM logZ'])
        # utils.append_to_csv(filename, [CALIs, logZ])
        return CALIs
    else:
        raise("Algorthim not defined")


def plot_result(H_a, MU):
    for alg in ['GBR_ibound=10','GBR_ibound=20', 'MF', 'BP']:
        data = utils.read_csv("{}_seattle_CALI_init_inf=['V0']_H_a={}_MU={}.csv".format(alg, H_a, MU))[1:-1]
        CALI = [float(row[-1]) for row in data]
        plt.plot(range(len(CALI)), CALI, label=alg.replace('_ibound=',''))
    plt.title(r"$\mu$={}, $H_a$={}".format(MU, H_a), fontsize=12)
    plt.xlabel('node number')
    plt.ylabel('CALI')
    plt.legend()
    plt.savefig("./results/MU={}_H_a={}.png".format(MU, H_a))
    plt.show()

def compare_subplots():
    HS = [1e-2,5e-2,1e-1]
    MUS = [1e-4,2e-4,4e-4,6e-4]

    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True)
    # files = os.listdir('./results')
    for alg in ['GBR_ibound=20','GBR_ibound=5','GBR_ibound=10', 'MF', 'BP']:
        for i in range(len(HS)):
            for j in range(len(MUS)):
                data = utils.read_csv("{}_seattle_CALI_init_inf=['V0']_H_a={}_MU={}.csv".format(alg, HS[i], MUS[j]))[1:-1]
                CALI = [float(row[-1]) for row in data]
                # print(CALI)
                plt.subplot(4,3,i+j*len(HS)+1)
                # if alg == 'GBR_ibound=20': alg = 'Exact'
                plt.plot(range(len(CALI)), CALI, label=alg.replace('_ibound=',''))
                plt.title(r"$\mu$={}, $H_a$={}".format(MUS[j], HS[i]), fontsize=8)

                # quit()
    plt.legend(loc='upper right')
    fig.text(0.5, 0.04, r"$H_a$", ha='center')
    fig.text(0.04, 0.5, r"$\mu$", va='center', rotation='vertical')
    plt.show()

def MF_1_subplots():
    HS = [1.0]
    MUS = [1e-4,2e-4,4e-4,6e-4]

    # fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
    # files = os.listdir('./results')
    for alg in ['GBR_ibound=20']:
        for i in range(len(HS)):
            for j in range(len(MUS)):
                data = utils.read_csv("{}_seattle_CALI_init_inf=['V0']_H_a={}_MU={}.csv".format(alg, HS[i], MUS[j]))[1:-1]
                CALI = [float(row[-1]) for row in data]
                # print(CALI)
                # plt.subplot(4,1,i+j*len(HS)+1)
                # if alg == 'GBR_ibound=20': alg = 'Exact'
                plt.plot(range(len(CALI)), CALI, label=r"$\mu$={}".format(MUS[j]))


                # quit()
    plt.title(r"$H_a$={} using GBR with ibound 20".format(HS[0]))
    plt.legend(loc='upper right')
    plt.xlabel('node number')
    plt.ylabel('CALI')
    # plt.text(0.5, 0.04, r"$H_a$", ha='center')
    # plt.text(0.04, 0.5, r"$\mu$", va='center', rotation='vertical')
    plt.show()



def CALI_vs_node_number(HS, MUS, init_inf):
    for alg in ['BE']: # ,'GBR_ibound=20'
        for i in range(len(HS)):
            filename = "CALI_vs_node_number_initinf={}_H={}.csv".format(init_inf[0]+1, HS[i])
            utils.append_to_csv(filename, ['CALIs'])
            utils.append_to_csv(filename, list(range(1,11)))
            for j in range(len(MUS)):
                CALI = implement(case = 'seattle', alg = 'GBR', init_inf = init_inf, H_a = HS[i], MU = MUS[j], ibound = 20)
                utils.append_to_csv(filename, [CALI])

                # for idx,val in enumerate(CALI):

                plt.plot( range(len(CALI)), CALI,'*-')
                # continue
                # meanCALI.append(np.mean(CALI))
                # plt.plot(MUS[j], np.mean(CALI),'*', label=r"$\mu$={}".format(MUS[j]))
            # plt.plot(MUS, meanCALI)
            # plt.show()
            plt.title(r"H_a={}, infected node {}".format(HS[i], ith_object_name('V',init_inf[0]+1)))
            plt.xticks(range(0, len(CALI)))
            plt.xlabel(r"node number")
            plt.ylabel('CALI')
            plt.axis([0, len(CALI)-1, -1.1, 1.1])
            plt.savefig("./results/CALI_vs_node_number_initinf={}_MU_range[{},{}]_H={}.png".format(init_inf[0]+1,MUS[0],MUS[-1], HS[i]))
            # plt.show()
        plt.clf()

def generate_data_for(H_a, MU, init_inf=[0]):
    implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = H_a, MU = MU)
    # implement(case = 'seattle', alg = 'MF', init_inf = [0], H_a = H_a, MU = MU)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 5)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 10)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 20)
    implement(case = 'seattle', alg = 'BE', init_inf = init_inf, H_a = H_a, MU = MU)

def CALI_vs_MU(config):
    HS = list(config.keys())
    data = {}
    for alg in ['BP','GBR20']:
        data[alg] = {}
        ibound=10
        if 'GBR' in alg: ibound = int(alg[-2:])
        for H in HS:
            for inf in range(20):
                data[alg][str((H,inf+1))] = {}
                for MU in config[H]:
                    print("here")
                    CALI = implement(case = 'seattle', alg = alg[:3], init_inf = [inf], H_a = H, MU = MU, ibound = ibound)
                    print("done")
                    data[alg][str((H,inf+1))][MU] = CALI
                    plt.plot(MU*np.ones(len(CALI)), CALI, '*')
                plt.title(r"H_a={}, infected node {}".format(H, ith_object_name('V',inf+1)))
                plt.xlabel(r"$\mu$")
                plt.ylabel('CALI')
                plt.axis([config[H][0], config[H][-1], -1.1, 1.1])
                plt.savefig("./results/CALIvsMU_alg_{}_H={}_MUrange=[{},{}]_initinf={}.png".format(alg,H, config[H][0], config[H][-1], inf+1))
                plt.clf()

    with open("./results/CALI_vs_MU.json", 'w') as outfile:
        json.dump(data, outfile)

# number of MU
N = 10
mu = 6e-5
mus = np.round(np.linspace(0,mu, N),5)
config = {key: mus for key in [0.02, 0.05, 0.1, 0.2, 0.5]}
# CALI_vs_MU(config)
CALI = implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = mu, ibound = 50)
print(CALI)
