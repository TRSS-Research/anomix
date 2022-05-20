from .univariate import (
    BetaMixtureModel,
    BinomialMixtureModel,
    CauchyMixtureModel,
    ExponentialMixtureModel,
    GeometricMixtureModel,
    LogNormalMixtureModel,
    NormalMixtureModel,
    NormalLocationMixtureModel,
    NormalMixtureModelSklearn,
    PoissonMixtureModel,
    StudentsTMixtureModel,
    ZeroInflatedBinomialMixtureModel,
    ZeroInflatedExponentialMixtureModel,
    ZeroInflatedNormalMixtureModel,
    ZeroInflatedPoissonMixtureModel,
    ZetaMixtureModel
)

from ..utils.static import DISTRIBUTIONS

model_dict = {
    BetaMixtureModel.__name__: BetaMixtureModel,
    BinomialMixtureModel.__name__: BinomialMixtureModel,
    CauchyMixtureModel.__name__: CauchyMixtureModel,
    ExponentialMixtureModel.__name__: ExponentialMixtureModel,
    GeometricMixtureModel.__name__: GeometricMixtureModel,
    LogNormalMixtureModel.__name__: LogNormalMixtureModel,
    NormalMixtureModel.__name__: NormalMixtureModel,
    NormalMixtureModelSklearn.__name__: NormalMixtureModelSklearn,
    NormalLocationMixtureModel.__name__:NormalLocationMixtureModel,
    PoissonMixtureModel.__name__: PoissonMixtureModel,
    StudentsTMixtureModel.__name__: StudentsTMixtureModel,
    ZeroInflatedBinomialMixtureModel.__name__: ZeroInflatedBinomialMixtureModel,
    ZeroInflatedExponentialMixtureModel.__name__: ZeroInflatedExponentialMixtureModel,
    ZeroInflatedNormalMixtureModel.__name__: ZeroInflatedNormalMixtureModel,
    ZeroInflatedPoissonMixtureModel.__name__: ZeroInflatedPoissonMixtureModel,
    ZetaMixtureModel.__name__: ZetaMixtureModel
}


UNIVARIATE = [BetaMixtureModel, BinomialMixtureModel, CauchyMixtureModel, ExponentialMixtureModel,
              GeometricMixtureModel, LogNormalMixtureModel, NormalMixtureModel, PoissonMixtureModel,
              StudentsTMixtureModel, ZeroInflatedBinomialMixtureModel, ZeroInflatedExponentialMixtureModel,
              ZeroInflatedNormalMixtureModel, ZeroInflatedPoissonMixtureModel, ZetaMixtureModel]


IMPLEMENTED = UNIVARIATE

ExternalModelNameDict = {DISTRIBUTIONS.NORMAL: NormalMixtureModelSklearn,
                         DISTRIBUTIONS.LOGNORMAL: NormalMixtureModelSklearn,
                         DISTRIBUTIONS.POISSON: PoissonMixtureModel,
                         DISTRIBUTIONS.STUDENTST: StudentsTMixtureModel,
                         DISTRIBUTIONS.ZIPOISSON: ZeroInflatedPoissonMixtureModel,
                         DISTRIBUTIONS.ZINORMAL: ZeroInflatedNormalMixtureModel,
                         DISTRIBUTIONS.EXPONENTIAL: ExponentialMixtureModel,
                         DISTRIBUTIONS.ZIEXPONENTIAL: ZeroInflatedExponentialMixtureModel,
                         DISTRIBUTIONS.CAUCHY: CauchyMixtureModel,
                         DISTRIBUTIONS.BINOMIAL: BinomialMixtureModel,
                         DISTRIBUTIONS.GEOMETRIC: GeometricMixtureModel,
                         DISTRIBUTIONS.BETA: BetaMixtureModel,
                         }
