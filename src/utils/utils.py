

def trunc_normal_(x, mean=0., std=1.):
  "Truncated normal initialization (approximation)"
  # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
  return x.normal_().fmod_(2).mul_(std).add_(mean)