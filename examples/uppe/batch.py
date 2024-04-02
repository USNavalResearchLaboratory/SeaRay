import sys
sys.path.append('modules')
import rays
import examples.uppe.air_fil as fil
import importlib

# MUST be copied to and run from root directory
# e.g. `python batch.py run device=titan`

# Example of an UPPE batch job.
# Examines effect of varying chi3.
# The diagnostics are written by appending a suffix to `base filename`,
# but one can as easily have them go to different directories.

num_runs = 3

for irun in range(num_runs):
    # N.b. in general we must reload the input file each iteration
    importlib.reload(fil)
    chi3 = fil.helper.chi3(1.0,'5e-24 m2/W')
    fil.optics[1]['chi3'] = chi3*float(irun)
    fil.diagnostics['base filename'] = 'out/test' + str(irun)
    print()
    print()
    print("*********************************")
    print("This is batch job",irun+1,"of",num_runs)
    print("*********************************")
    print()
    print()
    rays.run(sys.argv[2:],fil.sim,fil.sources,fil.optics,fil.diagnostics)