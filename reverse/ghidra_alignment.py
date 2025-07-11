import os
import logging
import csv
from ghidra.app.util.headless import HeadlessScript
from ghidra.app.decompiler import DecompInterface

# Get script arguments and determine the save folder
argv = getScriptArgs()

try:
    if len(argv) == 2:
        output_folder = argv[0]
        results_folder = argv[1]
    elif len(argv) == 1:
        output_folder = argv[0]
        results_folder = os.path.join(output_folder, 'results')
    elif len(argv) == 0:
        output_folder = os.getcwd()
        results_folder = os.path.join(os.getcwd(), 'results')
    else:
        raise ValueError("Invalid number of arguments")
except Exception as e:
    error_message = "An error occurred while setting parameters: {}".format(e)
    logging.error(error_message, exc_info=True)

program_name = currentProgram.getName()
program_folder = results_folder

# Create the program-specific directory
if not os.path.exists(program_folder):
    os.makedirs(program_folder)

# Set up logging
log_file_path = os.path.join(output_folder, 'extraction.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

csv_file_path = os.path.join(program_folder, program_name + '.csv')

try:
    fm = currentProgram.getFunctionManager()
    funcs = fm.getFunctions(True)
    
    # Initialize the decompiler
    decompiler = DecompInterface()
    decompiler.openProgram(currentProgram)

    all_data = []
    
    for func in funcs:
        if func.isExternal() or func.isThunk():
            continue
            
        entry_point = func.getEntryPoint()
        name = func.getName()
        
        decompile_results = decompiler.decompileFunction(func, 30, None)
        hign_func = decompile_results.getHighFunction()
        if hign_func is None:
            logging.warning("Decompilation failed for function: {}".format(name))
            continue

        for pcode_op in hign_func.getPcodeOps():
            addr = str(pcode_op.getSeqnum().getTarget())
            operation = str(pcode_op)
            opcode = pcode_op.getMnemonic()
            
            all_data.append([name, addr, operation, opcode])

    with open(csv_file_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Function', 'Address', 'Operation', 'Opcode'])
        writer.writerows(all_data)

    decompiler.closeProgram()
    logging.info("Successfully extracted CSV data for {}".format(program_name))

except Exception as e:
    error_message = "An error occurred while writing the files: {}".format(e)
    logging.error(error_message, exc_info=True)
    decompiler.closeProgram()