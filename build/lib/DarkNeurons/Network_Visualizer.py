import netron

def network_visualizer(model):

	print('---We are using Open Source Library Netron for Network Visualization---')
	print('--- Supoort Netron at https://github.com/lutzroeder/netron ---')

	netron.start(model)
