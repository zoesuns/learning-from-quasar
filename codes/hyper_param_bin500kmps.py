import os
os.environ['MPLCONFIGDIR'] = './'
import time
import subprocess
import torch
import sbi
from sbi import utils
from sbi import analysis
from sbi import inference
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi

from sbi.analysis import check_sbc, run_sbc, get_nltp, sbc_rank_plot
from sbi.inference import SNPE, SNPE_C, prepare_for_sbi, simulate_for_sbi
from sbi.simulators import linear_gaussian, diagonal_linear_gaussian


theta = torch.load('training_theta_bin500kmps.t')
x = torch.load('training_x_sm50_bin500kmps.t')
    
theta_cal = torch.load('calib_theta_bin500kmps.t')
x_cal = torch.load('calib_x_bin500kmps.t')

# set prior distribution for the parameters
num_dim = 2
prior = utils.BoxUniform(low=torch.tensor([0,0]), high=torch.tensor([1,8000]))

par_list=[(5,5), (5,10),(5,20),
          (10,5),(10,10),(10,20),
          (20,5),(20,10),(20,20)
         ]
for par in par_list:
    # instantiate the neural density estimator
    model="maf"
    nhf=int(par[0])
    nt=int(par[1])
    nb=int(2)
    neural_posterior = utils.posterior_nn(
        model=model,  hidden_features=nhf, num_transforms=nt,num_blocks=nb
    )
    
    # setup the inference procedure with the SNPE-C procedure
    inference = SNPE(prior=prior, density_estimator=neural_posterior)
    
    inference.append_simulations(theta, x)
    
    t0=time.time()
    density_estimator = inference.train()
    t1=time.time()
    print("time spent = {:.1f} min".format((t1-t0)/60.))
    with open("bin500kmps_maf_{:0d}_{:0d}_{:0d}.txt".format(nhf,nt,nb),"w") as fi:
        fi.write("time spent = {:.1f} min\n".format((t1-t0)/60.))
    
    posterior = inference.build_posterior(density_estimator)
    
    
    dirname="bin500kmps_maf_{:0d}_{:0d}_{:0d}/".format(nhf,nt,nb)
    subprocess.call(["mkdir",dirname])
    for i in range(50):
        samples = posterior.sample((10000,), x=torch.tensor(x_cal[i]))
        fig, ax = analysis.pairplot(
            samples.to("cpu"),
            points=theta_cal[i],
            labels=["xHI", r"wp"],
            limits=[[0, 1], [0, 8000]],
            points_colors="r",
            points_offdiag={"markersize": 6},
            figsize=(5, 5),
        )
        fig.savefig(dirname+str(i)+".png")
        plt.close(fig)
    
    num_posterior_samples = 1000
    ranks, dap_samples = run_sbc(
        theta_cal, x_cal, posterior, num_posterior_samples=num_posterior_samples, show_progress_bar = False
    )
    
    check_stats = check_sbc(
        ranks, theta_cal, dap_samples, num_posterior_samples=num_posterior_samples
    )
    print(
        f"kolmogorov-smirnov p-values \ncheck_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}"
    )
    with open("bin500kmps_maf_{:0d}_{:0d}_{:0d}.txt".format(nhf,nt,nb),"a") as fi:
        fi.write(f"kolmogorov-smirnov p-values \ncheck_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}")
    
    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type="hist",
        num_bins=None,  # by passing None we use a heuristic for the number of bins.
    )
    f.savefig("bin500kmps_rank_hist_maf_{:0d}_{:0d}_{:0d}.pdf".format(nhf,nt,nb))
    f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf")
    f.savefig("bin500kmps_rank_cdf_maf_{:0d}_{:0d}_{:0d}.pdf".format(nhf,nt,nb))
