import { useState, useEffect } from "react";
import {
  Package,
  Plus,
  Search,
  Edit,
  Trash2,
  AlertTriangle,
  X,
  Filter,
  Box,
  CheckCircle2,
  AlertCircle
} from "lucide-react";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { productsAPI } from "../lib/api";
import { formatCurrency, formatNumber } from "../lib/utils";
import { toast } from "sonner";

// 🕒 دالة لتنسيق التاريخ والوقت
const formatDateTime = (dateString) => {
  if (!dateString) return "";
  return new Date(dateString).toLocaleString("ar-SY", {
    year: "numeric",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true
  });
};

const initialFormData = {
  name: "",
  purchase_price: "",
  sale_price: "",
  quantity: "",
  min_quantity: "10",
  description: "",
};

export default function Products() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [showLowStock, setShowLowStock] = useState(false);
  
  // حالات النوافذ
  const [dialogOpen, setDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  
  const [editingProduct, setEditingProduct] = useState(null);
  const [productToDelete, setProductToDelete] = useState(null);
  
  const [formData, setFormData] = useState(initialFormData);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchProducts();
  }, [showLowStock]);

  const fetchProducts = async () => {
    try {
      const response = await productsAPI.getAll(showLowStock);
      setProducts(response.data);
    } catch (error) {
      toast.error("فشل في تحميل المنتجات");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      const data = {
        name: formData.name,
        purchase_price: parseFloat(formData.purchase_price),
        sale_price: parseFloat(formData.sale_price),
        quantity: parseInt(formData.quantity),
        min_quantity: parseInt(formData.min_quantity),
        description: formData.description || null,
      };

      if (editingProduct) {
        await productsAPI.update(editingProduct.id, data);
        toast.success("تم تحديث المنتج بنجاح");
      } else {
        await productsAPI.create(data);
        toast.success("تم إضافة المنتج بنجاح");
      }

      setDialogOpen(false);
      setEditingProduct(null);
      setFormData(initialFormData);
      fetchProducts();
    } catch (error) {
      toast.error(editingProduct ? "فشل في تحديث المنتج" : "فشل في إضافة المنتج");
    } finally {
      setSubmitting(false);
    }
  };

  const handleEdit = (product) => {
    setEditingProduct(product);
    setFormData({
      name: product.name,
      purchase_price: product.purchase_price.toString(),
      sale_price: product.sale_price.toString(),
      quantity: product.quantity.toString(),
      min_quantity: product.min_quantity.toString(),
      description: product.description || "",
    });
    setDialogOpen(true);
  };

  // دالة فتح نافذة تأكيد الحذف
  const confirmDelete = (product) => {
    setProductToDelete(product);
    setDeleteDialogOpen(true);
  };

  // دالة تنفيذ الحذف الفعلي
  const executeDelete = async () => {
    if (!productToDelete) return;
    
    try {
      await productsAPI.delete(productToDelete.id);
      toast.success("تم حذف المنتج بنجاح");
      fetchProducts();
      setDeleteDialogOpen(false);
      setProductToDelete(null);
    } catch (error) {
      toast.error("فشل في حذف المنتج");
    }
  };

  const filteredProducts = products.filter((product) =>
    product.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const isLowStock = (product) => product.quantity <= product.min_quantity;

  return (
    <div className="space-y-8 animate-fadeIn pb-10 font-sans">
      
      {/* 1. Header */}
      <div className="relative overflow-hidden rounded-3xl bg-slate-900 p-8 text-white shadow-2xl ring-1 ring-white/10">
        <div className="absolute top-0 left-0 h-full w-full bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-purple-500 blur-[100px] opacity-20"></div>
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-indigo-500 blur-[100px] opacity-20"></div>
        
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">المنتجات</h1>
            <p className="text-slate-400 max-w-lg">
              إدارة المخزون، الأسعار، وتفاصيل المنتجات في مكان واحد.
            </p>
          </div>
          
          <Button 
            onClick={() => setDialogOpen(true)} 
            className="bg-white text-slate-900 hover:bg-slate-100 font-bold px-6 h-12 rounded-xl shadow-lg transition-transform hover:scale-105"
            data-testid="add-product-btn"
          >
            <Plus className="h-5 w-5 ml-2" />
            إضافة منتج جديد
          </Button>
        </div>
      </div>

      {/* 2. Filters */}
      <Card className="border-none shadow-sm bg-white rounded-2xl overflow-hidden">
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-4 items-center">
            <div className="relative flex-1 w-full">
              <Search className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
              <Input
                placeholder="البحث عن منتج بالاسم..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pr-12 h-12 bg-slate-50 border-slate-100 focus:bg-white transition-colors rounded-xl text-base"
                data-testid="search-products"
              />
            </div>
            
            {/* زر الفلترة مع تصحيح مشكلة اختفاء النص */}
            <Button
              variant={showLowStock ? "default" : "outline"}
              onClick={() => setShowLowStock(!showLowStock)}
              className={`h-12 px-6 rounded-xl border-slate-200 transition-colors ${
                showLowStock 
                  ? "bg-amber-500 hover:bg-amber-600 text-white border-transparent shadow-md shadow-amber-200" 
                  : "bg-white text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`}
              data-testid="filter-low-stock"
            >
              <Filter className="h-4 w-4 ml-2" />
              {showLowStock ? "إظهار الكل" : "عرض النواقص فقط"}
              {showLowStock && <AlertTriangle className="h-4 w-4 mr-2" />}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 3. Table */}
      <Card className="border-none shadow-sm bg-white overflow-hidden rounded-3xl">
        <CardContent className="p-0">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="h-10 w-10 rounded-full border-4 border-slate-200 border-t-indigo-600 animate-spin"></div>
            </div>
          ) : filteredProducts.length > 0 ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader className="bg-slate-50/80">
                  <TableRow className="hover:bg-transparent border-slate-100">
                    <TableHead className="text-right py-5 px-6 font-bold text-slate-600">اسم المنتج</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">سعر الشراء</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">سعر البيع</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الكمية</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">تاريخ الإضافة</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الحالة</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الإجراءات</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredProducts.map((product) => (
                    <TableRow 
                      key={product.id} 
                      className="hover:bg-indigo-50/30 transition-colors border-slate-50 group hover:-translate-y-1 hover:shadow-lg duration-300 cursor-pointer"
                      data-testid={`product-row-${product.id}`}
                    >
                      <TableCell className="font-medium px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="h-10 w-10 rounded-xl bg-indigo-50 flex items-center justify-center text-indigo-600 shadow-sm group-hover:scale-110 group-hover:bg-indigo-500 group-hover:text-white transition-all duration-300">
                            <Box className="h-5 w-5" />
                          </div>
                          <div>
                            <p className="font-bold text-slate-800 text-base group-hover:text-indigo-900 transition-colors">{product.name}</p>
                            {product.description && (
                              <p className="text-xs text-slate-400 line-clamp-1">{product.description}</p>
                            )}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="py-4">
                        <span className="text-slate-600 font-medium bg-slate-50 px-2 py-1 rounded-lg border border-slate-100 group-hover:border-indigo-100 transition-colors">
                          {formatCurrency(product.purchase_price)}
                        </span>
                      </TableCell>
                      <TableCell className="py-4">
                        <span className="text-slate-900 font-bold group-hover:text-indigo-700 transition-colors">
                          {formatCurrency(product.sale_price)}
                        </span>
                      </TableCell>
                      <TableCell className="py-4">
                        <div className="flex items-center gap-2">
                          <span className={`text-base font-bold ${isLowStock(product) ? "text-amber-600" : "text-slate-700"}`}>
                            {formatNumber(product.quantity)}
                          </span>
                          <span className="text-xs text-slate-400 bg-slate-50 px-1.5 py-0.5 rounded">
                             / {product.min_quantity}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-slate-500 text-sm py-4 dir-ltr text-right">
                        {formatDateTime(product.created_at)}
                      </TableCell>
                      <TableCell className="py-4">
                        {isLowStock(product) ? (
                          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold bg-amber-50 text-amber-700 border border-amber-100 group-hover:bg-amber-100 transition-colors">
                            <AlertTriangle className="h-3 w-3 ml-1" />
                            مخزون منخفض
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-100 group-hover:bg-emerald-100 transition-colors">
                            <CheckCircle2 className="h-3 w-3 ml-1" />
                            متوفر
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="py-4">
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => { e.stopPropagation(); handleEdit(product); }}
                            className="h-8 w-8 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-all hover:scale-110"
                            data-testid={`edit-product-${product.id}`}
                          >
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => { e.stopPropagation(); confirmDelete(product); }}
                            className="h-8 w-8 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all hover:scale-110"
                            data-testid={`delete-product-${product.id}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <div className="h-20 w-20 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                <Package className="h-10 w-10 text-slate-300" />
              </div>
              <h3 className="text-lg font-bold text-slate-900">لا توجد منتجات</h3>
              <p className="text-slate-500 max-w-sm mt-2">
                {searchTerm ? "لم يتم العثور على نتائج مطابقة لبحثك." : "لم تقم بإضافة أي منتجات بعد. أضف منتجك الأول لتبدأ."}
              </p>
              {!searchTerm && (
                <Button onClick={() => setDialogOpen(true)} className="mt-6 bg-slate-900 text-white hover:bg-slate-800 shadow-lg hover:shadow-xl transition-all hover:-translate-y-1">
                  <Plus className="h-4 w-4 ml-2" />
                  إضافة أول منتج
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
              {editingProduct ? "تعديل بيانات المنتج" : "إضافة منتج جديد"}
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-5 mt-4">
            <div>
              <Label htmlFor="name" className="text-slate-600 mb-1.5 block">اسم المنتج</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
                className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                data-testid="product-name-input"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="purchase_price" className="text-slate-600 mb-1.5 block">سعر الشراء</Label>
                <Input
                  id="purchase_price"
                  type="number"
                  step="0.01"
                  min="0"
                  value={formData.purchase_price}
                  onChange={(e) => setFormData({ ...formData, purchase_price: e.target.value })}
                  required
                  className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="product-purchase-price-input"
                />
              </div>
              <div>
                <Label htmlFor="sale_price" className="text-slate-600 mb-1.5 block">سعر البيع</Label>
                <Input
                  id="sale_price"
                  type="number"
                  step="0.01"
                  min="0"
                  value={formData.sale_price}
                  onChange={(e) => setFormData({ ...formData, sale_price: e.target.value })}
                  required
                  className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="product-sale-price-input"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="quantity" className="text-slate-600 mb-1.5 block">الكمية الحالية</Label>
                <Input
                  id="quantity"
                  type="number"
                  min="0"
                  value={formData.quantity}
                  onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                  required
                  className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="product-quantity-input"
                />
              </div>
              <div>
                <Label htmlFor="min_quantity" className="text-slate-600 mb-1.5 block">الحد الأدنى للتنبيه</Label>
                <Input
                  id="min_quantity"
                  type="number"
                  min="0"
                  value={formData.min_quantity}
                  onChange={(e) => setFormData({ ...formData, min_quantity: e.target.value })}
                  required
                  className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="product-min-quantity-input"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="description" className="text-slate-600 mb-1.5 block">الوصف (اختياري)</Label>
              <Input
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                data-testid="product-description-input"
              />
            </div>
            
            {/* أزرار النافذة مع إصلاح الألوان */}
            <DialogFooter className="gap-3 sm:justify-start">
              <Button type="submit" disabled={submitting} className="h-11 px-8 rounded-xl bg-slate-900 text-white hover:bg-slate-800 shadow-md" data-testid="submit-product-btn">
                {submitting ? "جاري الحفظ..." : editingProduct ? "حفظ التعديلات" : "إضافة المنتج"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setDialogOpen(false);
                  setEditingProduct(null);
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

      {/* نافذة تأكيد الحذف */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="max-w-sm rounded-3xl p-6" dir="rtl">
          <div className="flex flex-col items-center text-center gap-4 pt-2">
            <div className="h-14 w-14 rounded-full bg-red-100 flex items-center justify-center mb-2">
              <AlertCircle className="h-8 w-8 text-red-600" />
            </div>
            <DialogHeader className="space-y-2">
              <DialogTitle className="text-xl font-bold text-slate-900 text-center">تأكيد الحذف</DialogTitle>
              <div className="text-slate-500 text-sm max-w-xs mx-auto">
                هل أنت متأكد أنك تريد حذف المنتج <span className="font-bold text-slate-900">"{productToDelete?.name}"</span>؟
                <br />
                لا يمكن التراجع عن هذا الإجراء لاحقاً.
              </div>
            </DialogHeader>
            
            {/* أزرار الحذف والإلغاء مع إصلاح الألوان */}
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