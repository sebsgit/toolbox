#!/usr/bin/env python3

import json
import argparse
import re
import subprocess
import os


def is_std_allocator(stack_entry):
    return 'std::make_shared' in stack_entry['function'] \
        or 'std::make_unique' in stack_entry['function']

def is_excluded(stack_entry, parse_config):
    filename = stack_entry['file']
    for exclude_entry in parse_config.get('ignore_files', []):
        if filename.startswith(exclude_entry):
            if parse_config.get('keep_std_allocators', False) and is_std_allocator(stack_entry):
                return False
            return True
    return False


def strip_function_info(function_info):
    function_info = re.sub(r'#\d+\s+', '', function_info)
    function_info = re.sub(r'0x[0-9a-fA-F]+\s+in\s+', '', function_info)
    function_info = re.sub(r'\(.*\)', '', function_info)
    return function_info.strip()


def parse_gdb_line(line):
    split = line.split(' at ')
    if len(split) == 2:
        result = {}
        file_info = split[1].strip()
        function_info = split[0].strip()
        result['function'] = strip_function_info(function_info)
        result['file'] = file_info.split(':')[0]
        result['line'] = int(file_info.split(':')[1])
        return result
    return None


def parse_stacktrace(input_file, parse_config):
    stacktrace = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            entry = parse_gdb_line(line)
            if entry and not is_excluded(entry, parse_config):
                stacktrace.append(entry)
    return stacktrace


def generate_vs_code_commands(stacktrace):
    unique_files = {}
    for entry in stacktrace:
        if entry['file'] not in unique_files:
            unique_files[entry['file']] = entry['line']

    commands = []
    for entry in unique_files.keys():
        command = f'code -g {entry}:{unique_files[entry]} -r'
        commands.append(command)
    return commands


def open_vs_code(commands, config):
    commands.reverse()
    first = True
    for command in commands:
        subprocess.run(command.split(' '))
        if first:
            first = False
            first_wait = str(config.get('initial_vscode_delay_sec', '2'))
            print(f'Waiting for vscode to wake up for {first_wait} secs...')
            subprocess.run(['sleep', first_wait])
        else:
            subprocess.run(['sleep', '1'])


def generate_graphviz_content(stacktrace):
    graph = 'digraph G {\n'
    call_seq = []
    for entry in stacktrace:
        call_seq.append(f'{entry["function"]}')
    for i in range(len(call_seq) - 1, 0, -1):
        graph += '"{}" -> "{}"\n'.format(call_seq[i], call_seq[i-1])
    graph += '}'
    return graph


def generate_graphviz_image(dot_content, config):
    dot_file = os.path.join(config['tempdir'], "graph.dot")
    with open(dot_file, 'w') as f:
        f.write(dot_content)
    print("Generating Graphviz DOT file: {}".format(dot_file))
    subprocess.run(['dot', '-Tpng', dot_file, '-o', '{}.png'.format('graph')])


def generate_gdb_breakpoint_section(funcname, br_count, out_file):
    br_counter = '$count_{}'.format(br_count)
    br = 'set {} = 0\n'.format(br_counter)
    br += 'br {}\n'.format(funcname)
    br += 'commands\n'
    br += '\teval "set logging file {}%i", {}++\n\t'.format(out_file, br_counter)
    br += '\n\t'.join(['set logging overwrite on', 'set logging enabled on', 'bt', 'set logging enabled off', 'continue'])
    br += '\nend\n'

    return br


def generate_gdb_stacktrace_filename(config, breakpoint):
    breakpoint = re.sub(r'\W+', '_', breakpoint)
    return '{}_{}'.format(config['stacktrace_filename'], breakpoint)


def generate_gbd_script_content(config):
    gdb = 'file {}\n'.format(config['program'])
    if config['args']:
        gdb += 'set args {}\n'.format(' '.join(str(s) for s in config['args']))
    br_count = 0
    for br in config['breakpoints']:
        gdb += generate_gdb_breakpoint_section(br, br_count, generate_gdb_stacktrace_filename(config, br))
        br_count += 1
    gdb += 'run\nquit\n'
    return gdb


def run_debug_session(config):
    content = generate_gbd_script_content(config)
    gdb_script_file = os.path.join(config['tempdir'], 'script.gdb')
    with open(gdb_script_file, 'w') as f:
        f.write(content)
    print("Running debug session using gdb script: {}".format(gdb_script_file))
    subprocess.run(['gdb', '--batch', '--command=' + gdb_script_file])

def validate_args(args):
    if args.config and not args.config[0].endswith('.json'):
        raise argparse.ArgumentError(f'Config file must have a.json extension: {args.config}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs=1, help='Config JSON file')
    parser.add_argument('--viz', nargs=1, help='Generate graphviz output for the given stacktrace')
    parser.add_argument('--code', nargs=1, help='Open the stacktrace files in the vs code')
    
    args = parser.parse_args()

    validate_args(args)

    with open(args.config[0], 'r') as f:
        config = json.load(f)

    if args.viz:
        print("Generating graphviz output for : {}".format(args.viz))
        trace = parse_stacktrace(args.viz[0], config)
        gviz = generate_graphviz_content(trace)
        generate_graphviz_image(gviz, config['debug_session'])
        return

    if args.code:
        print("Opening stacktrace files in VS Code: {}".format(args.code))
        trace = parse_stacktrace(args.code[0], config)
        print("Extracted stacktrace:")
        for entry in trace:
            print(str(entry))
        commands = generate_vs_code_commands(trace)
        open_vs_code(commands, config)
        return

    run_debug_session(config['debug_session'])

if __name__ == '__main__':
    main()
