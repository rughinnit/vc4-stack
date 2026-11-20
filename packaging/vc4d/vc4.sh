# vc4-stack environment (shipped by vc4d, depended on by vc4-toolkit).
#
# Makes the unprivileged daemon backend the default for every shell, so a
# normal user does not need root or /dev/mem to run vc4infer, vc4-query or
# vc4train.  This changes no clocks, no governor and no voltage.

# helpers installed by the stack (nothing here today, but the layout is fixed)
case ":${PATH}:" in
*:/usr/lib/vc4infer/bin:*) ;;
*) PATH="/usr/lib/vc4infer/bin:${PATH}" ;;
esac
export PATH

# Every client goes through /run/vc4d.sock.  Set VC4INFER_DAEMON=0 explicitly
# to use the root//dev/mem path instead.
export VC4INFER_DAEMON=1
