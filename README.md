
# ArcFace face recognition

Bu proje, ArcFace tabanlı bir yüz tanıma modelinin eğitimini ve testini gerçekleştirmek için geliştirilmiştir. SE (Squeeze-and-Excitation) ve IR (Identity Residual) bloklarını içeren farklı konfigürasyonlarla eğitilebilir. Eğitim sırasında çeşitli optimizasyon teknikleri ve öğrenme oranı düzenlemeleri kullanılır.


## İçindekiler

- [Proje Tanımı](#proje-tanımı)
- [Kurulum](#kurulum)
- [Konfigürasyon](#konfigürasyon)
- [Veri Yükleme](#Veri-Yükleme)
- [Model Tanımı](#Model-Tanımı)
- [Eğitim](#Eğitim)
- [Değerlendirme](#Değerlendirme)
- [Kontrol Noktalarını Kaydetme](#Kontrol-Noktalarını-Kaydetme)



## Proje Tanımı  
Bu proje Bilgisayarlı Görü Uygulama Alanları isimli bir BTK akademi kursunda ArcFace ile yüz tanıma konu başlığı altında yazılmıştır. Kodlara açıklamalar ve detaylı bir readme dosyası eklenmiştir. ArcFace tabanlı bir yüz tanıma modelini eğitmek ve test etmek için geliştirilmiştir. Model, SE (Squeeze-and-Excitation) ve IR (Identity Residual) blokları kullanılarak yapılandırılabilir. Eğitim sürecinde çeşitli optimizasyon teknikleri ve öğrenme oranı düzenlemeleri uygulanır.

## Kurulum
Zip dosyasını bilgisayarınıza github üzerinden indirebilirsiniz. Daha sonra Google Colab, Jupyter notebook veya dilediğiniz bir derleyiciden çalıştırabilirsiniz. Dosyayı google colab'te açmak için drive'a yüklemek gerek ve baştaki " from google.colab import drive
drive.mount('/content/drive') " kısmı gerekli. Diğer derleyicilerde çalıştırmak için bu kısma gerek yok silebilirsiniz.
Gerekli kütüphaneleri indirmek için:
pip install tqdm easydict tensorflow torch torchvision
 
## Konfigürasyon
Konfigürasyon ayarları, çeşitli parametrelerin kolayca yönetilmesini sağlayan bir EasyDict nesnesinde saklanır. Bu ayarlar, eğitim ve değerlendirme süreçlerini kontrol eder. Aşağıda bazı önemli konfigürasyon parametreleri verilmiştir:

**train_root:** Eğitim verilerinin bulunduğu dizin.  
**lfw_test_root:** LFW test verilerinin bulunduğu dizin.  
**lfw_file_list:** LFW veri dosyalarının listesi.  
**mode:** Kullanılacak modelin türü (örneğin 'se_ir' veya 'ir').(  
*Squeeze-and-Excitation Blokları:('se_ir')* Özellik haritalarının önem derecelerini öğrenir ve kanal bazında ağırlıklar uygular. Bu, modelin hangi özelliklerin daha önemli olduğunu belirlemesine yardımcı olur.  
*Identity Residual Blokları:('ir')* Geriye doğru geçiş (skip connection) sağlayarak, öğrenmeyi daha verimli hale getirir ve derin ağların eğitimini kolaylaştırır
)  
**depth:** Modelin derinliği (örneğin 50 katmanlı bir model).    
Derinlik arttıkça modelin daha karmaşık özellikleri öğrenme kapasitesi artar. Ancak, derinlik arttıkça hesaplama maliyeti ve overfitting (aşırı uyum) riski de artar. Daha derin modeller genellikle daha yüksek doğruluk sağlar ancak eğitim ve test süresi artar.  
**margin_type:** Kullanılacak marj tipi (örneğin 'ArcFace' marj fonksiyonu).  
**feature_dim:** Özellik vektörünün boyutu.  
**scale_size:** Ölçek faktörü.  
**batch_size:** Her eğitim adımında kullanılacak örnek sayısı.  
**lr:** Öğrenme oranı.  
**milestones:** Öğrenme oranının düşürüleceği dönemler.  
**total_epoch:** Toplam eğitim dönemi sayısı.  
**save_folder:** Modellerin kaydedileceği dizin.  
**device:** Eğitim için kullanılacak cihaz (CPU veya GPU).  
**num_workers:** Veri yükleyici için kullanılacak işçi sayısı.  
**pin_memory:** Bellek pinleme.

## Veri Yükleme ve İşleme
Veri yükleme ve işleme adımları, modelin doğru bir şekilde eğitilmesi için kritik öneme sahiptir. Bu adımlar, verilerin doğru formatta olmasını ve modelin etkili bir şekilde öğrenmesini sağlamak için yapılır. Projede veri yükleme ve işleme aşamaları şu şekilde gerçekleşir:

### Veri Dönüşümü

Veri dönüşümü, ham verilerin modelin öğrenme süreci için uygun hale getirilmesini sağlar. Dönüşüm işlemleri şu adımları içerir:

- **Piksel Normalizasyonu:** Resimler, [0, 255] aralığından [0.0, 1.0] aralığına dönüştürülür. Bu, modelin daha hızlı ve verimli bir şekilde öğrenmesini sağlar.
- **Normalizasyon:** Resimlerin her kanalının (kırmızı, yeşil, mavi) ortalamasını ve standart sapmasını kullanarak normalizasyon yapılır. Bu, verilerin istatistiksel olarak daha dengeli hale gelmesini sağlar ve modelin eğitim sürecinde daha istikrarlı performans göstermesine yardımcı olur.

Örnek dönüşüm kodu:


import torchvision.transforms as trans

transform = trans.Compose([
    trans.ToTensor(),  # Piksel değerlerini [0,255] aralığından [0.0,1.0] aralığına dönüştürür
    trans.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalizasyon
]) 

**Eğitim Veri Yükleyici**  
Eğitim veri yükleyici, eğitim verilerini modelin öğrenme süreci için hazırlar. Eğitim veri yükleyici, veri kümesini belirli bir batch boyutunda yükler ve verilerin GPU'ya taşınmasını sağlar. Eğitim veri kümesi, get_train_loader fonksiyonu ile yüklenir. Bu fonksiyon, veri kümesini belirli bir batch boyutunda ve veri işleme adımlarını içerecek şekilde yapılandırır.  
**Kod örneği:**  
trainloader, class_num = get_train_loader(conf)  
print('number of id:', class_num)  # Sınıf sayısını yazdırıyoruz  

**LFW Test Veri Yükleyici**  
LFW (Labeled Faces in the Wild) test veri yükleyici, test verilerini modelin değerlendirilmesi için hazırlar. Test veri yükleyici, verileri belirli bir batch boyutunda yükler ve verilerin GPU'ya taşınmasını sağlar. Test veri kümesi, torch.utils.data.DataLoader sınıfı kullanılarak yüklenir.  
**Kod örneği:**  
lfwdataset = LFW(conf.lfw_test_root, conf.lfw_file_list, transform=transform)  
lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=128, num_workers=conf.num_workers)

**Veri Ön İşleme**  
Veri ön işleme, veri kümesinin eğitim ve test süreçleri için uygun hale getirilmesini sağlar. Bu aşama, veri kalitesini artırmak ve modelin genelleme yeteneğini geliştirmek için önemlidir. Veri ön işleme adımları şunları içerebilir:

**Resimlerin Yeniden Boyutlandırılması:** Resimlerin boyutlarını belirli bir ölçüye getirmek, modelin daha iyi öğrenmesini sağlar.  
**Veri Artırma:** Eğitim verilerini çeşitli dönüşümlerle (dönme, kesme, vb.) genişletmek, modelin genelleme yeteneğini artırır.  
**Düşük Kaliteli Verilerin Temizlenmesi:** Verilerin kalitesini artırmak için düşük kaliteli veya hatalı veriler temizlenir veya çıkarılır.  
Veri yükleme ve işleme işlemleri, modelin etkili bir şekilde öğrenmesini ve değerlendirilmesini sağlamak için kritik adımlardır. Bu adımların doğru bir şekilde uygulanması, modelin performansını ve doğruluğunu doğrudan etkiler.
## Model Tanımı

Model tanımı, sinir ağı mimarisi ve marj katmanını içerir. Modelin performansını etkileyen bu bileşenler, dikkatlice yapılandırılmalıdır. Aşağıda modelin çeşitli bileşenleri hakkında detaylı bilgiler verilmiştir.

### Ağ ve Marj Tanımı

Modelin ağ yapısı ve marj fonksiyonu, modelin derinliğini ve özellik boyutunu belirleyerek tanımlanır. Bu aşamada:

- **Sinir Ağı Mimarisi:** Model, SE (Squeeze-and-Excitation) ve IR (Identity Residual) bloklarını içerebilir. SE blokları, özellik haritalarının önem derecelerini öğrenir ve kanal bazında ağırlıklar uygular, bu da modelin hangi özelliklerin daha önemli olduğunu belirlemesine yardımcı olur. IR blokları ise geri geçiş (skip connection) sağlayarak, ağın derinliğini artırır ve eğitim sürecini daha verimli hale getirir.

- **Marj Katmanı:** ArcFace algoritması, özellik vektörleri ile sınıf merkezleri arasındaki açıları hesaplar. Bu marj katmanı, yüz tanıma görevlerinde daha yüksek doğruluk sağlar ve modelin performansını artırır.

### Kayıp Fonksiyonu

Kayıp fonksiyonu, modelin tahminlerinin gerçek etiketlerle ne kadar uyumlu olduğunu ölçen bir işlevdir. Bu projede kullanılan kayıp fonksiyonu şu şekildedir:

- **CrossEntropyLoss:** Çoklu sınıflı sınıflandırma problemlerinde yaygın olarak kullanılan bir kayıp fonksiyonudur. Bu fonksiyon, modelin tahminleri ile gerçek etiketler arasındaki farkı hesaplar ve modelin öğrenme sürecini yönlendirir. Eğitim sırasında bu fonksiyon, modelin doğruluğunu artırmak için optimize edilir.

### Optimizatör

Optimizatör, modelin ağırlıklarını güncelleyerek öğrenme sürecini yönetir. Bu projede kullanılan optimizatör ve ayarları şunlardır:

- **Stochastic Gradient Descent (SGD):** SGD, modelin ağırlıklarını güncellemek için kullanılan bir optimizasyon yöntemidir. Momentum ve ağırlık çürümesi gibi ek özellikler ile birlikte kullanılır, bu da modelin daha hızlı ve daha etkili bir şekilde öğrenmesini sağlar. SGD'nin öğrenme oranı ve diğer hiperparametreleri, modelin eğitim sürecini optimize etmek için dikkatlice ayarlanır.
## Eğitim

Eğitim süreci, modelin performansını artırmak için öğrenme oranı zamanlaması ve çoklu dönemlerle yürütülür. Bu süreç, verilerin işlenmesi, modelin eğitilmesi ve kayıpların hesaplanmasını içerir.

### Öğrenme Oranı Zamanlama

Öğrenme oranı, modelin eğitim sürecinde belirli dönemlerde azaltılır. Bu strateji, modelin daha iyi genelleme yapmasını sağlar ve aşırı uyum riskini azaltır. Öğrenme oranını azaltma işlemi, genellikle eğitim sürecinin belirli dönemlerinde gerçekleştirilir.

### Eğitim Döngüsü

Eğitim döngüsü, modelin eğitim sürecini yöneten bir dizi adımdan oluşur. Her bir epoch (dönem) sırasında:

- **Veri Yükleme:** Eğitim verileri yüklenir.
- **Model Eğitimi:** Model, yüklenen verilerle eğitilir. Gradienler hesaplanır ve modelin ağırlıkları güncellenir.
- **Değerlendirme:** Modelin performansı, eğitim sürecinin her döneminde değerlendirilir.
- **Kontrol Noktaları:** Modelin durumunu kaydedilerek en iyi performans gösteren modeller saklanır.

**Örnek Eğitim Döngüsü:**

1. Model eğitim moduna alınır.
2. Öğrenme oranı belirli dönemlerde azaltılır.
3. Eğitim verileri üzerinde iterasyon yapılır.
4. Modelin tahminleri ile gerçek etiketler arasındaki kayıplar hesaplanır.
5. Model değerlendirilir ve doğruluk hesaplanır.
6. En iyi performans gösteren model kontrol noktaları kaydedilir.

Bu süreç, modelin genel doğruluğunu ve performansını artırmaya yönelik bir dizi adımı içerir. Eğitim sırasında modelin öğrenme süreci düzenli olarak izlenir ve gerekli iyileştirmeler yapılır.
## Değerlendirme

Her dönem sonunda modelin performansını izlemek ve doğruluğunu ölçmek için test işlemi gerçekleştirilir. Bu değerlendirme, modelin LFW (Labeled Faces in the Wild) veri seti üzerinde yapılır. Değerlendirme süreci şu adımları içerir:

1. **Modeli Eval Moduna Alma:** Modelin doğruluğunu değerlendirmek için modelin "eval" moduna alınması gerekir. Bu mod, modelin eğitimde kullanılan dropout gibi belirli davranışlarını devre dışı bırakır ve modelin performansını daha doğru bir şekilde ölçer.

2. **Doğruluk Hesaplama:** LFW veri seti üzerindeki doğruluk, modelin tahminlerinin gerçek etiketlerle ne kadar uyumlu olduğunu belirler. Doğruluk, modelin gerçek dünyadaki performansını değerlendirmek için kullanılır.

3. **En İyi Doğruluk Güncelleme:** Modelin doğruluğu hesaplandıktan sonra, en iyi doğruluk değeri güncellenir. Bu, eğitim sürecinde modelin en iyi performansını izlemeyi sağlar ve modelin iyileştirilmesine yönelik kararlar almaya yardımcı olur.

**Değerlendirme Süreci:**

- Model "eval" moduna alınır.
- LFW veri seti üzerinde modelin tahminleri alınır.
- Doğruluk hesaplanır.
- En iyi doğruluk değeri güncellenir ve saklanır.## Kontrol Noktalarını Kaydetme

Kontrol noktaları, modelin eğitim sürecindeki belirli dönemlerdeki durumunu kaydeder ve en iyi performans gösteren modellerin saklanmasını sağlar. Bu, eğitim sürecinde modelin en iyi durumunu korumak ve gerektiğinde geri dönmek için önemlidir. Kontrol noktalarını kaydetme süreci şu adımları içerir:

1. **Kontrol Noktası Kaydetme:** Her dönem sonunda modelin ağırlıkları ve durum bilgileri kaydedilir. Bu, eğitim sırasında modelin belirli bir noktadaki durumunun yedeklenmesini sağlar.

2. **En İyi Modeli İzleme:** En iyi doğruluk değerini gösteren model ayrı olarak izlenir. Bu model, eğitim sürecinde en yüksek performansı gösteren model olarak kabul edilir ve özel olarak saklanır.

3. **Kontrol Noktalarının Saklanması:** Kaydedilen kontrol noktaları, modelin eğitim sürecinde ilerlemeyi ve performansı takip etmeye yardımcı olur. Eğitim tamamlandıktan sonra, bu kontrol noktaları modelin gelecekteki kullanımları için de saklanabilir.

**Kontrol Noktalarını Kaydetme Süreci:**

- Her dönem sonunda modelin durumu kaydedilir.
- En iyi performansı gösteren model ayrı olarak saklanır.
- Kontrol noktaları, modelin eğitim sürecinde ilerlemeyi izlemek ve gerektiğinde geri dönmek için kullanılır.
