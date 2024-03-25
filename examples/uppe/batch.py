import sys
sys.path.append('modules')
import examples.uppe.air_fil as fil
import rays

# MUST be copied to and run from root directory

# Example of an UPPE batch job.
# Examines effect of varying chi3.
# The diagnostics are written by appending a suffix to `base filename`,
# but one can as easily have them go to different directories.

num_runs = 3
chi3 = fil.helper.chi3(1.0,'5e-24 m2/W')
vol = None

for i in range(len(fil.optics)):
    if fil.optics[i]['object'].name == 'air':
        vol = i
if vol==None:
    raise NameError('could not find volume object')

for irun in range(num_runs):
    fil.optics[vol]['chi3'] = chi3*float(irun)
    fil.diagnostics['base filename'] = 'out/test' + str(irun)
    print()
    print()
    print("*********************************")
    print("This is batch job",irun+1,"of",num_runs)
    print("*********************************")
    print()
    print()
    rays.run(sys.argv[2:],fil.sim,fil.ray,fil.wave,fil.optics,fil.diagnostics)