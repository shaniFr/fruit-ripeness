import wx
from test_MyFruit import MyImage


class MyFrame(wx.Frame):
    def __init__(self):
        self.my_frame = MyImage()
        super().__init__(parent=None, title='My Fruit Program')
        panel = wx.Panel(self)
        my_sizer = wx.BoxSizer(wx.VERTICAL)
        lbl = wx.StaticText(panel, style=wx.ALIGN_CENTRE)
        self.text_ctrl1 = wx.TextCtrl(panel, style=wx.TE_CENTRE)
        font = wx.Font(20, wx.ROMAN, wx.ITALIC, wx.NORMAL)
        lbl.SetFont(font)
        lbl.SetLabel("My Fruit")
        check_btn = wx.Button(panel, label='Check my fruit!')
        check_btn.Bind(wx.EVT_BUTTON, self.on_press_check_fruit)
        self.text_ctrl1.Bind(wx.EVT_LEFT_DOWN, self.set_image_path)
        my_sizer.Add(lbl, 0, wx.ALL | wx.CENTER, 10)
        my_sizer.Add(self.text_ctrl1, 0, wx.ALL | wx.EXPAND, 10)
        my_sizer.Add(check_btn, 10, wx.ALL | wx.CENTER, 5)

        panel.SetSizer(my_sizer)
        self.Show()
        self.clear()

    def on_press_choose_paths(self, event):
        # self.my_frame.set_paths()
        self.text_ctrl1.SetValue(self.my_frame.get_image_path())

    def on_press_check_fruit(self, event):
        self.my_frame.checkImage()
        self.clear()

    def set_image_path(self, event):
        self.my_frame.set_image_path()
        self.text_ctrl1.SetValue(self.my_frame.get_image_path())

    def clear(self):
        self.my_frame.clear()
        self.text_ctrl1.SetValue("Click to select image")
