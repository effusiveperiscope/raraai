#!/bin/bash
# Make the project that /etc/vast_boot.d/36-sync-workspace.sh copies out of
# /opt/workspace-internal writable by the instance's non-root `user` account.
#
# That stage copies with `cp -ru`, which creates the destination files as root
# with the source mode masked by the boot umask: a 0777 directory in the image
# lands as 0755, and `user` then cannot create checkpoints_stage1/,
# checkpoints_stage2/, logs_stage1/, logs_stage2/ or data/ inside it.  Fixing
# the modes after the copy is cheaper and more robust than trying to encode them
# in the image.
#
# This file is SOURCED by boot_default.sh (see its "stages are SOURCED" comment
# next to the /etc/vast_boot.d loop), so it must not `set -e`, `exit`, or leave
# variables behind that could shadow the boot shell's.
__project="${WORKSPACE:-/workspace}/25_transformer"
if [ -d "${__project}" ]; then
    chmod -R a+rwX "${__project}" 2>/dev/null || true
fi
# The model/tokenizer/NLTK caches live outside the workspace (so they are not
# copied into a volume) and are written at build time as root; make them
# writable too, or a `user`-account session cannot pull an extra model.
for __cache in /opt/hf-cache /opt/nltk_data /opt/torch-cache; do
    [ -d "${__cache}" ] && chmod -R a+rwX "${__cache}" 2>/dev/null || true
done
unset __project __cache
