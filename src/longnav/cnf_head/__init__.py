"""Replacement policy head: differentials -> encoder -> z -> conditional flow -> decoder.

Stage one (this package) is the *unconditional* action autoencoder plus the
reconstruction gate that decides whether the design is viable at all.

The load-bearing idea, restated so it is not lost:

    A continuous density on R^n cannot put mass on the data's exact-zero atom.
    The pushforward of a continuous density through a NON-INJECTIVE decoder can.

So the decoder's output nonlinearity is a soft threshold with a learned per-dim
width, `sign(x) * relu(|x| - tau)`, which has a genuinely flat region: a
positive-volume set of latents maps to *exactly* zero action. A smooth MLP would
give near-zeros instead, which is the creeping bug relocated one layer down.

Modules
-------
`data`            per-tick differentials, cache building, channel scaling
`models`          SoftThreshold, ActionAutoencoder
`train_ae`        training entry point (wandb online)
`gate`            the reconstruction gate + figures
`export_latents`  latent dataset export for the flow stage
"""
