from data import get_all_data
from utils import save

records = get_all_data()
save(records, "records")
