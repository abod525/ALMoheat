import { useState, useEffect } from "react";
import {
  Users,
  Plus,
  Search,
  Edit,
  Trash2,
  Phone,
  Mail,
  MapPin,
  User,
  Briefcase,
  Building2,
  Filter,
  ArrowRight,
  AlertCircle
} from "lucide-react";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Badge } from "../components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { contactsAPI } from "../lib/api";
import { formatCurrency } from "../lib/utils";
import { toast } from "sonner";

const initialFormData = {
  name: "",
  contact_type: "customer",
  phone: "",
  email: "",
  address: "",
};

export default function Contacts() {
  const [contacts, setContacts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTab, setActiveTab] = useState("all");
  
  // حالات النوافذ
  const [dialogOpen, setDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  
  const [editingContact, setEditingContact] = useState(null);
  const [contactToDelete, setContactToDelete] = useState(null);
  
  const [formData, setFormData] = useState(initialFormData);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchContacts();
  }, [activeTab]);

  const fetchContacts = async () => {
    try {
      const type = activeTab === "all" ? null : activeTab;
      const response = await contactsAPI.getAll(type);
      setContacts(response.data);
    } catch (error) {
      toast.error("فشل في تحميل العملاء");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      const data = {
        ...formData,
        phone: formData.phone || null,
        email: formData.email || null,
        address: formData.address || null,
      };

      if (editingContact) {
        await contactsAPI.update(editingContact.id, data);
        toast.success("تم تحديث جهة الاتصال بنجاح");
      } else {
        await contactsAPI.create(data);
        toast.success("تم إضافة جهة الاتصال بنجاح");
      }

      setDialogOpen(false);
      setEditingContact(null);
      setFormData(initialFormData);
      fetchContacts();
    } catch (error) {
      toast.error(editingContact ? "فشل في تحديث جهة الاتصال" : "فشل في إضافة جهة الاتصال");
    } finally {
      setSubmitting(false);
    }
  };

  const handleEdit = (contact) => {
    setEditingContact(contact);
    setFormData({
      name: contact.name,
      contact_type: contact.contact_type,
      phone: contact.phone || "",
      email: contact.email || "",
      address: contact.address || "",
    });
    setDialogOpen(true);
  };

  const confirmDelete = (contact) => {
    setContactToDelete(contact);
    setDeleteDialogOpen(true);
  };

  const executeDelete = async () => {
    if (!contactToDelete) return;

    try {
      await contactsAPI.delete(contactToDelete.id);
      toast.success("تم حذف جهة الاتصال بنجاح");
      fetchContacts();
      setDeleteDialogOpen(false);
      setContactToDelete(null);
    } catch (error) {
      toast.error("فشل في حذف جهة الاتصال");
    }
  };

  const filteredContacts = contacts.filter((contact) =>
    contact.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // ✅ تم التعديل هنا: قلبنا "له" و "عليه" لتصبح صحيحة محاسبياً
  const getBalanceDisplay = (balance) => {
    if (balance > 0) {
      // رصيد موجب = دين على العميل
      return { text: `عليه: ${formatCurrency(balance)}`, class: "text-emerald-700 bg-emerald-50 border-emerald-100" };
    } else if (balance < 0) {
      // رصيد سالب = المحل مدين للعميل (له)
      return { text: `له: ${formatCurrency(Math.abs(balance))}`, class: "text-red-700 bg-red-50 border-red-100" };
    }
    return { text: "—", class: "text-slate-400 bg-slate-50 border-slate-100" };
  };

  const totalCustomers = contacts.filter(c => c.contact_type === 'customer').length;
  const totalSuppliers = contacts.filter(c => c.contact_type === 'supplier').length;

  return (
    <div className="space-y-8 animate-fadeIn pb-10 font-sans">
      
      {/* 1. Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-slate-900 p-8 text-white shadow-2xl ring-1 ring-white/10">
        <div className="absolute top-0 left-0 h-full w-full bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-500 blur-[100px] opacity-20"></div>
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-blue-500 blur-[100px] opacity-20"></div>
        
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">العملاء</h1>
            <p className="text-slate-400 max-w-lg">
              إدارة قاعدة بيانات العملاء والموردين ومتابعة أرصدتهم المالية.
            </p>
          </div>
          
          <Button 
            onClick={() => setDialogOpen(true)} 
            className="bg-white text-slate-900 hover:bg-slate-100 font-bold px-6 h-12 rounded-xl shadow-lg transition-transform hover:scale-105"
            data-testid="add-contact-btn"
          >
            <Plus className="h-5 w-5 ml-2" />
            إضافة جهة اتصال
          </Button>
        </div>
      </div>

      {/* 2. Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* إجمالي العملاء */}
        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-1">إجمالي العملاء</p>
                <h3 className="text-3xl font-bold text-slate-900">{totalCustomers}</h3>
              </div>
              <div className="h-14 w-14 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600 group-hover:bg-emerald-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-emerald-200">
                <User className="h-7 w-7" />
              </div>
            </div>
            <div className="mt-4 flex items-center text-xs font-medium text-emerald-700 bg-emerald-50 w-fit px-2 py-1 rounded-lg">
              زبائن نشطون
            </div>
          </CardContent>
        </Card>

        {/* إجمالي الموردين */}
        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-1">إجمالي الموردين</p>
                <h3 className="text-3xl font-bold text-slate-900">{totalSuppliers}</h3>
              </div>
              <div className="h-14 w-14 rounded-2xl bg-blue-50 flex items-center justify-center text-blue-600 group-hover:bg-blue-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-blue-200">
                <Briefcase className="h-7 w-7" />
              </div>
            </div>
            <div className="mt-4 flex items-center text-xs font-medium text-blue-700 bg-blue-50 w-fit px-2 py-1 rounded-lg">
              شركاء العمل
            </div>
          </CardContent>
        </Card>

        {/* الكل */}
        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-1">المجموع الكلي</p>
                <h3 className="text-3xl font-bold text-slate-900">{contacts.length}</h3>
              </div>
              <div className="h-14 w-14 rounded-2xl bg-indigo-50 flex items-center justify-center text-indigo-600 group-hover:bg-indigo-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-indigo-200">
                <Users className="h-7 w-7" />
              </div>
            </div>
            <div className="mt-4 flex items-center text-xs font-medium text-indigo-700 bg-indigo-50 w-fit px-2 py-1 rounded-lg">
              قاعدة البيانات
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3. Filters & Search */}
      <Card className="border-none shadow-sm bg-white rounded-2xl overflow-hidden">
        <CardContent className="p-4">
          <Tabs value={activeTab} onValueChange={setActiveTab} dir="rtl" className="w-full">
            <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
              <TabsList className="bg-slate-100 p-1 h-12 rounded-xl">
                <TabsTrigger value="all" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm transition-all">
                  الكل
                </TabsTrigger>
                <TabsTrigger value="customer" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-emerald-700 data-[state=active]:shadow-sm transition-all">
                  العملاء
                </TabsTrigger>
                <TabsTrigger value="supplier" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-blue-700 data-[state=active]:shadow-sm transition-all">
                  الموردين
                </TabsTrigger>
              </TabsList>

              <div className="relative w-full sm:w-72">
                <Search className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  placeholder="بحث عن اسم، هاتف..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pr-12 h-12 bg-slate-50 border-slate-100 focus:bg-white transition-colors rounded-xl text-base"
                  data-testid="search-contacts"
                />
              </div>
            </div>
          </Tabs>
        </CardContent>
      </Card>

      {/* 4. Contacts Table */}
      <Card className="border-none shadow-sm bg-white overflow-hidden rounded-3xl">
        <CardContent className="p-0">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="h-10 w-10 rounded-full border-4 border-slate-200 border-t-blue-600 animate-spin"></div>
            </div>
          ) : filteredContacts.length > 0 ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader className="bg-slate-50/80">
                  <TableRow className="hover:bg-transparent border-slate-100">
                    <TableHead className="text-right py-5 px-6 font-bold text-slate-600">الاسم</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">النوع</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">بيانات الاتصال</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الرصيد</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الإجراءات</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredContacts.map((contact) => {
                    const balanceInfo = getBalanceDisplay(contact.balance);
                    return (
                      <TableRow 
                        key={contact.id} 
                        className="hover:bg-indigo-50/30 transition-colors border-slate-50 group cursor-pointer hover:-translate-y-1 hover:shadow-lg duration-300"
                        data-testid={`contact-row-${contact.id}`}
                      >
                        <TableCell className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className={`h-10 w-10 rounded-xl flex items-center justify-center shadow-sm transition-colors ${contact.contact_type === 'customer' ? 'bg-emerald-50 text-emerald-600 group-hover:bg-emerald-200' : 'bg-blue-50 text-blue-600 group-hover:bg-blue-200'}`}>
                              {contact.contact_type === 'customer' ? (
                                <User className="h-5 w-5" />
                              ) : (
                                <Briefcase className="h-5 w-5" />
                              )}
                            </div>
                            <div>
                              <p className="font-bold text-slate-800 text-base group-hover:text-slate-900 transition-colors">{contact.name}</p>
                              {contact.address && (
                                <div className="flex items-center gap-1 text-xs text-slate-400 mt-0.5">
                                  <MapPin className="h-3 w-3" />
                                  <span>{contact.address}</span>
                                </div>
                              )}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell className="py-4">
                          <Badge variant="secondary" className={`px-2.5 py-1 rounded-lg transition-colors ${contact.contact_type === 'customer' ? 'bg-emerald-50 text-emerald-700 group-hover:bg-emerald-100' : 'bg-blue-50 text-blue-700 group-hover:bg-blue-100'}`}>
                            {contact.contact_type === 'customer' ? 'عميل' : 'مورد'}
                          </Badge>
                        </TableCell>
                        <TableCell className="py-4">
                          <div className="flex flex-col gap-1">
                            {contact.phone ? (
                              <div className="flex items-center gap-2 text-sm text-slate-600 group-hover:text-slate-800 transition-colors">
                                <Phone className="h-3.5 w-3.5 text-slate-400" />
                                <span className="dir-ltr">{contact.phone}</span>
                              </div>
                            ) : (
                              <span className="text-xs text-slate-400">-</span>
                            )}
                            {contact.email && (
                              <div className="flex items-center gap-2 text-xs text-slate-500 group-hover:text-slate-700 transition-colors">
                                <Mail className="h-3.5 w-3.5 text-slate-400" />
                                <span>{contact.email}</span>
                              </div>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="py-4">
                          <span className={`inline-block px-3 py-1 rounded-lg text-sm font-bold border ${balanceInfo.class}`}>
                            {balanceInfo.text}
                          </span>
                        </TableCell>
                        <TableCell className="py-4">
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={(e) => { e.stopPropagation(); handleEdit(contact); }}
                              className="h-8 w-8 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all hover:scale-110"
                              data-testid={`edit-contact-${contact.id}`}
                            >
                              <Edit className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={(e) => { e.stopPropagation(); confirmDelete(contact); }}
                              className="h-8 w-8 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all hover:scale-110"
                              data-testid={`delete-contact-${contact.id}`}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <div className="h-20 w-20 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                <Users className="h-10 w-10 text-slate-300" />
              </div>
              <h3 className="text-lg font-bold text-slate-900">لا توجد جهات اتصال</h3>
              <p className="text-slate-500 max-w-sm mt-2">
                {searchTerm ? "لم يتم العثور على نتائج مطابقة." : "لم تقم بإضافة أي عملاء أو موردين بعد."}
              </p>
              {!searchTerm && (
                <Button onClick={() => setDialogOpen(true)} className="mt-6 bg-slate-900 text-white hover:bg-slate-800 shadow-lg hover:shadow-xl transition-all hover:-translate-y-1">
                  <Plus className="h-4 w-4 ml-2" />
                  إضافة أول جهة اتصال
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add/Edit Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-md rounded-3xl" dir="rtl">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-slate-900">
              {editingContact ? "تعديل بيانات" : "إضافة جهة اتصال"}
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-5 mt-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="col-span-2">
                <Label htmlFor="name" className="text-slate-600 mb-1.5 block">الاسم الكامل</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="contact-name-input"
                />
              </div>
              <div className="col-span-2">
                <Label htmlFor="contact_type" className="text-slate-600 mb-1.5 block">التصنيف</Label>
                <Select
                  value={formData.contact_type}
                  onValueChange={(value) => setFormData({ ...formData, contact_type: value })}
                >
                  <SelectTrigger className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400" data-testid="contact-type-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="customer">عميل (زبون)</SelectItem>
                    <SelectItem value="supplier">مورد (تاجر)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="phone" className="text-slate-600 mb-1.5 block">رقم الهاتف</Label>
                <Input
                  id="phone"
                  value={formData.phone}
                  onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                  className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="contact-phone-input"
                />
              </div>
              <div>
                <Label htmlFor="email" className="text-slate-600 mb-1.5 block">البريد الإلكتروني</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="contact-email-input"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="address" className="text-slate-600 mb-1.5 block">العنوان</Label>
              <Input
                id="address"
                value={formData.address}
                onChange={(e) => setFormData({ ...formData, address: e.target.value })}
                className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                data-testid="contact-address-input"
              />
            </div>

            <DialogFooter className="gap-3 sm:justify-start">
              <Button type="submit" disabled={submitting} className="h-11 px-8 rounded-xl bg-slate-900 text-white hover:bg-slate-800 shadow-md hover:shadow-lg transition-all hover:-translate-y-0.5" data-testid="submit-contact-btn">
                {submitting ? "جاري الحفظ..." : editingContact ? "حفظ التعديلات" : "إضافة"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setDialogOpen(false);
                  setEditingContact(null);
                  setFormData(initialFormData);
                }}
                className="h-11 px-6 rounded-xl text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900"
              >
                إلغاء
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* 🆕 نافذة تأكيد الحذف */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="max-w-sm rounded-3xl p-6" dir="rtl">
          <div className="flex flex-col items-center text-center gap-4 pt-2">
            <div className="h-14 w-14 rounded-full bg-red-100 flex items-center justify-center mb-2">
              <AlertCircle className="h-8 w-8 text-red-600" />
            </div>
            <DialogHeader className="space-y-2">
              <DialogTitle className="text-xl font-bold text-slate-900 text-center">تأكيد الحذف</DialogTitle>
              <div className="text-slate-500 text-sm max-w-xs mx-auto">
                هل أنت متأكد أنك تريد حذف <span className="font-bold text-slate-900">"{contactToDelete?.name}"</span>؟
                <br />
                لا يمكن التراجع عن هذا الإجراء لاحقاً.
              </div>
            </DialogHeader>
            
            <DialogFooter className="flex gap-2 w-full mt-4">
              <Button variant="outline" onClick={() => setDeleteDialogOpen(false)} className="flex-1 h-11 rounded-xl text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900">
                إلغاء
              </Button>
              <Button onClick={executeDelete} className="flex-1 h-11 rounded-xl bg-red-600 hover:bg-red-700 text-white shadow-md shadow-red-200">
                حذف نهائي
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

    </div>
  );
}