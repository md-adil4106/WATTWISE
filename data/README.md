# WATTWISE Data Directory

This directory stores raw and processed electricity consumption datasets. Raw datasets are excluded from Git tracking via `.gitignore` to prevent committing large files.

## Dataset: Individual Household Electric Power Consumption

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Direct Download:** [household_power_consumption.zip](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)
- **Local File Placement:**
  Place the extracted file directly into this directory:
  ```
  data/household_power_consumption.txt
  ```

### Specifications
- **Observations:** 2,075,259 measurements gathered between December 2006 and November 2010 (47 months).
- **Sampling Rate:** 1-minute sampling interval.
- **Key Columns:**
  - `Date`: Date in format `dd/mm/yyyy`
  - `Time`: Time in format `hh:mm:ss`
  - `Global_active_power`: Household global minute-averaged active power (in kW)
  - `Global_reactive_power`: Household global minute-averaged reactive power (in kW)
  - `Voltage`: Minute-averaged voltage (in volts)
  - `Global_intensity`: Household global minute-averaged current intensity (in amperes)
  - `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`: Sub-metering energy (in watt-hour of active energy)
