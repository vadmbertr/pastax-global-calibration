from jaxtyping import Array, Float

from pastax.dynamics import LinearUV


class LinearDeterministic(LinearUV):
    def get_parameters(self) -> tuple[Float[Array, ""], Float[Array, ""]]:
        return {"intercept": self.intercept, "slope": self.slope}
