"""
The main testing script that connects to the client and server, runs the tests and collects data
"""
import pffrocd
import configparser
import time

"""Parse config file"""
config = configparser.ConfigParser()
config.read('config.ini')

client_ip = config.get('client', 'ip_address')
client_username = config.get('client', 'username')
client_key = config.get('client', 'private_ssh_key_path')
client_exec_path = config.get('client', 'executable_path')
client_exec_name = config.get('client', 'executable_name')
client_pffrocd_path = config.get('client', 'pffrocd_path')

server_ip = config.get('server', 'ip_address')
server_username = config.get('server', 'username')
server_key = config.get('server', 'private_ssh_key_path')
server_exec_path = config.get('server', 'executable_path')
server_exec_name = config.get('server', 'executable_name')
server_pffrocd_path = config.get('server', 'pffrocd_path')

test_mode = config.getboolean('misc', 'test_mode')
sec_lvl = config.getint('misc', 'security_level')
mt_alg = config.get('misc', 'mt_algorithm')
niceness = config.getint('misc', 'niceness')


def run_test():
    logger = pffrocd.setup_logging()

    # print all config options to the debug log
    logger.debug(f"pffrocd config: {pffrocd.get_config_in_printing_format(config)}")

    # get the list of people that have more than one image
    people = pffrocd.get_people_with_multiple_images(root_dir='lfw')

    if test_mode:
        # if debugging, only use one person
        people = people[:1]
        logger.info("RUNNING IN TEST MODE: only using one person")


    # run for all the people with multiple images
    for count, person in enumerate(people):
        # get all images from person
        imgs = pffrocd.get_images_in_folder(person)
        logger.info(f"Currently running for {person} ({count}/{len(people)})")
        logger.debug(f"Found {len(imgs)} images for {person}")

        # set the first image as the 'reference' image (registered at the service provider) and remove it from the list of images
        ref_img = imgs[0]
        imgs = imgs[1:]
        logger.debug(f"setting image as reference image: {ref_img}")

        # get as many images of other people as there are of that person
        other_imgs = pffrocd.get_random_images_except_person(root_dir='lfw', excluded_person=person, num_images=len(imgs))

        # join the two list of images together
        imgs = imgs + other_imgs

        # create shares of the reference image
        ref_img_embedding = pffrocd.get_embedding(ref_img)
        share0, share1 = pffrocd.create_shares(ref_img_embedding)

        # write the shares to the server and client
        pffrocd.write_share_to_remote_file(client_ip, client_username, client_key, f"{client_exec_path}/share1.txt", share0)
        pffrocd.write_share_to_remote_file(server_ip, server_username, server_key, f"{server_exec_path}/share0.txt", share1)

        # run the test for each image
        for img in imgs:
            logger.debug(f"Running test for {img}")

            # run the face embedding extraction script on the server
            stdout, stderr = pffrocd.execute_command(server_ip, server_username, f"{server_pffrocd_path}/env/bin/python {server_pffrocd_path}/pyscripts/extract_embedding.py -i {server_pffrocd_path}/{img} -o {server_exec_path}/embedding.txt", server_key)

            if stderr != '':
                logger.error("REMOTE EXECUTION OF COMMAND FAILED")
                logger.error(stderr)

            logger.debug(f"Embedding extracted by the server in {stdout} seconds")
            
            # send the files with embeddings to the client and server
            img_embedding = pffrocd.get_embedding(img)
            pffrocd.write_embeddings_to_remote_file(client_ip, client_username, client_key, f"{client_exec_path}/embeddings.txt", img_embedding, ref_img_embedding)
            pffrocd.write_embeddings_to_remote_file(server_ip, server_username, server_key, f"{server_exec_path}/embeddings.txt", img_embedding, ref_img_embedding)
            
            # run the sfe on both client and server in parallel
            logger.debug("Running sfe...")
            # stdout1, stderr1, stdout2, stderr2 = pffrocd.execute_command_parallel(host1=client_ip, username1=client_username, command1=f"{client_exec_path}/{client_exec_name} -r 1 -a {server_ip} -f {client_exec_path}/embeddings.txt", host2=server_ip, username2=server_username, command2=f"{server_exec_path}/{server_exec_name} -r 0 -a {server_ip} -f {server_exec_path}/embeddings.txt", private_key_path1=client_key, private_key_path2=server_key)
            # logger.debug("sfe done")
            # logger.debug(f"{stdout1=}")
            # logger.debug(f"{stderr1=}")
            # logger.debug(f"{stdout2=}")
            # logger.debug(f"{stderr2=}")

            command1 = f"cd {client_exec_path} ; nice -n {niceness} {client_exec_path}/{client_exec_name} -r 1 -a {server_ip} -f {client_exec_path}/embeddings.txt -s {sec_lvl} -x {mt_alg}"
            command2 = f"cd {server_exec_path} ; nice -n {niceness} {server_exec_path}/{server_exec_name} -r 0 -a {server_ip} -f {server_exec_path}/embeddings.txt -s {sec_lvl} -x {mt_alg}"
            start_time  = time.time()
            output = pffrocd.execute_command_parallel_alternative([client_ip, server_ip], client_username, "kamil123", command1, command2)
            total_time = time.time() - start_time
            logger.info(f"Finished! Total sfe time: {total_time} seconds")
            for host_output in output:
                hostname = host_output.host
                stdout = list(host_output.stdout)
                stderr = list(host_output.stderr)
                logger.debug("Host %s: exit code %s, output %s, error %s" % (
                    hostname, host_output.exit_code, stdout, stderr))

            # todo: save all results and timing data


            # todo: rerun the routine with powertop to gather energy consumption data


if __name__ == "__main__":
    run_test()