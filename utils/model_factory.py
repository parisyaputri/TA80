from models.digital_twin import DigitalTwin
from models.ddc import DynamicDeclarativeConstraints
from models.mv_arm import MVARMiner
from models.intelligent_body import IntelligentBody


def initialize_models():

    dt = DigitalTwin()

    ddc = DynamicDeclarativeConstraints()

    mv_arm = MVARMiner()

    ib = IntelligentBody(
        dt,
        ddc,
        mv_arm
    )

    return dt, ddc, mv_arm, ib