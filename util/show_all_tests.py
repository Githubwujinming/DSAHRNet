
import os
import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
from html import HTML
from util import mkdir
class HTMLCOMA(HTML):
    def __init__(self, web_dir, title, refresh=0):
        HTML.__init__(self, web_dir, title, refresh=refresh)
    
    def add_images(self, ims, txts, links, width=200):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

# 把一个数据集的各个模型测试结果保存到一个html页面中
if __name__ == '__main__':
    dataset = ['GD','LEVIR','SYSU','SV']
    methods = ['ChangeFormer','DSAMN','DSIFN','FCCDN','SNUN','CUBE']
    # methods = ['CUBE']
    txts = ['A','B','L']
    for m in methods:
        txts.append('%s_result'%m)
    web_dir = 'web_results'
    As = []
    Bs = []
    Ls = []
    comps = {}
    ind = 0
    for i in range(len(methods)):
        data = dataset[ind]
        dir = os.path.join(web_dir, data)
        imgs_name = sorted(os.listdir(dir+'/%s_%s'%(data,methods[i])))
        if len(As) == 0:
            As = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'A.png' in a]
            Bs = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'B.png' in a]
            Ls = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'L.png' in a]
        comps[methods[i]] = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'comp.png' in a]
    html = HTMLCOMA(os.path.join(web_dir,dataset[ind]),'%s_tests_results.html'%dataset[ind])
    html.add_header('%s_tests_results.html'%dataset[ind])
    for i in range(len(As)):
        ims = []
        A = As[i]
        B = Bs[i]
        L = Ls[i]
        comp = [comps[m][i] for m in methods]
        ims.extend((A,B,L))
        ims.extend(comp)
        txts[0] = A.split('.png')[0].split('/')[1]
        html.add_images(ims,txts,ims)
    html.save()